"""一键运行完整流程：

  1. 模拟“真实”经济（已知全部基本面，完全预见的水平解）；
  2. 生成研究者可观测的数据（t_obs 期截面 + 带抽样误差的历史面板）；
  3. 估计弹性：贸易弹性 θ（PPML 面板引力）、迁移弹性 ν（ACM/CDP 欧拉方程，2SLS），蒙特卡洛；
  4. 动态帽子代数（DHA）基线，并与水平解对照；
  5. 三个反事实（外国制造业生产率冲击、沿海—内陆交通走廊、跨地区迁移成本下降），
     分别用真实参数与估计参数求解，并与已知基本面的水平解对照；
  6. 输出图表（output/figures）、表格（output/tables）与结果摘要（output/results.md）。

用法：
  python run_pipeline.py            # 完整版（T = 400 季度，蒙特卡洛 100 次；约 5–10 分钟）
  python run_pipeline.py --quick    # 快速版（T = 200，蒙特卡洛 10 次）
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cdp import plotting
from cdp.dynamics import solve_dha_baseline, solve_dha_counterfactual, solve_levels_path, welfare
from cdp.estimation import estimate_migration_elasticity, estimate_trade_elasticity
from cdp.model import lm_reshape
from cdp.simulate import (build_economy, history_paths, initial_distribution, noisy_migration_panel,
                          noisy_trade_panel, observed_cross_section)
from cdp.static_eq import calibrate_iota

ROOT = Path(__file__).resolve().parent
MIN_FLOW = 3e-4          # 迁移回归只用样本期平均迁移份额 ≥ 该值的配对（见 estimation.py）


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# 反事实情景
# --------------------------------------------------------------------------------------
def make_scenarios(p, geo, T):
    """返回 {名称: dict(title, Ahat, kaphat, dtau_odds)}。t=0 宣布，此前未被预期。"""
    T1, N, J, M = T + 1, p.N, p.J, p.M
    manuf = p.sector_names.index("制造业")
    coast = np.flatnonzero(geo["coast"])
    inland = np.flatnonzero(~geo["coast"])

    # 1. 外国（F1）制造业生产率在 5 年（20 季度）内线性上升到 1.5 倍（类似“中国冲击”）
    Ahat = np.ones((T1, N, J))
    Ahat[:, p.n_dom, manuf] = np.exp(np.log(1.5) * np.minimum(np.arange(T1), 20) / 20)

    # 2. 沿海—内陆交通走廊：从 t=1 起双向贸易成本下降（商品 10%，服务 5%）
    kaphat = np.ones((T1, N, N, J))
    cut = np.array([0.9, 0.9, 0.95])
    for n in coast:
        for i in inland:
            kaphat[1:, n, i, :] = kaphat[1:, i, n, :] = cut

    # 3. 跨地区迁移成本下降：在价值不变时使跨地区迁移的相对概率（odds）乘以 1.65，
    #    即 Δτ = -ν·log(1.65)（效用单位）。t=0 的迁移决策即适用。
    reg = np.repeat(np.arange(p.n_dom), J + 1)
    cross = (reg[:, None] != reg[None]).astype(float)
    dtau_odds = np.broadcast_to(-np.log(1.65) * cross, (T1, M, M))

    return {
        "foreign_shock": dict(title="反事实 1：外国制造业生产率冲击（F1 制造业生产率 5 年内升至 1.5 倍）",
                              Ahat=Ahat, kaphat=None, dtau_odds=None),
        "corridor": dict(title="反事实 2：沿海—内陆交通走廊（双向贸易成本下降：商品 10%，服务 5%）",
                         Ahat=None, kaphat=kaphat, dtau_odds=None),
        "migration_reform": dict(title="反事实 3：跨地区迁移成本下降（户籍改革类政策，跨地区迁移几率 ×1.65）",
                                 Ahat=None, kaphat=None, dtau_odds=dtau_odds),
    }


def region_welfare(D0, L0, p):
    """按期初人口加权，把劳动力市场的终生价值变化汇总到地区，再换算为消费等价（%）。"""
    D = lm_reshape(D0, p)
    w = lm_reshape(L0, p)
    by_region = (D * w).sum(axis=1) / w.sum(axis=1)
    national = (D0 * L0).sum() / L0.sum()
    to_ce = lambda d: 100 * np.expm1((1 - p.beta) * d)
    return to_ce(by_region), float(to_ce(national))


# --------------------------------------------------------------------------------------
def main(quick=False, seed=7):
    out_fig = ROOT / "output" / "figures"
    out_tab = ROOT / "output" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)
    plotting.setup_style()

    t_obs = 40                                  # 10 年历史面板（季度）
    T = 200 if quick else 400                   # 反事实时域（季度）
    R = 10 if quick else 100                    # 蒙特卡洛次数
    trade_years = list(range(t_obs - 36, t_obs + 1, 4))   # 年度贸易数据

    # ---------- 1. 真实经济 ----------
    p, fund, geo = build_economy(seed=seed)
    L0 = initial_distribution(p)
    T_total = t_obs + T + 1
    hist = history_paths(fund, geo, p, t_obs, T_total)
    log(f"模拟经济：{p.N} 个地区（国内 {p.n_dom}）× {p.J} 个部门，{p.M} 个劳动力市场，{T_total} 期")
    truth = solve_levels_path(L0, fund, p, T_total, A_path=hist["A"], kappa_path=hist["kappa"])
    log(f"水平解收敛（{truth['iterations']} 次外层迭代）")
    obs = observed_cross_section(truth, fund, t_obs)
    moments = data_moments(truth, obs, p, t_obs)

    # ---------- 2–3. 估计（蒙特卡洛） ----------
    rng = np.random.default_rng(2026)
    rows = []
    for r in range(R):
        mig = noisy_migration_panel(truth, p, t_obs, rng, sigma_w=0.03)
        trade = noisy_trade_panel(truth, hist, trade_years, rng)
        th = estimate_trade_elasticity(trade)
        nu = estimate_migration_elasticity(mig, p.beta, min_flow=MIN_FLOW)
        row = dict(rep=r, nu_ols=nu["ols_fixed_beta"]["nu"], nu_iv=nu["iv_fixed_beta"]["nu"],
                   se_nu_iv=nu["iv_fixed_beta"]["se_nu"], first_stage_F=nu["iv_fixed_beta"]["first_stage_F"],
                   nu_free=nu["iv_free_beta"]["nu"], beta_free=nu["iv_free_beta"]["beta"],
                   n_obs_migration=nu["n_obs"], n_pairs_migration=nu["n_pairs"])
        for j, e in enumerate(th):
            row[f"theta_{j}"], row[f"se_theta_{j}"] = e["theta"], e["se"]
        rows.append(row)
    mc = pd.DataFrame(rows)
    mc.to_csv(out_tab / "estimation_mc.csv", index=False)
    est_summary = summarize_mc(mc, p)
    est_summary.to_csv(out_tab / "estimation_summary.csv", index=False)
    plotting.fig_estimation(mc, dict(nu=p.nu, theta=p.theta), p.sector_names, out_fig / "fig2_estimation_mc.png")
    log(f"估计完成（{R} 次蒙特卡洛）")

    # 研究者手中的一套估计值：第 0 次重复（一份数据）；ι 由观测数据反推
    first = mc.iloc[0]
    p_est = replace(p, nu=float(first["nu_iv"]), theta=np.array([first[f"theta_{j}"] for j in range(p.J)]),
                    iota=calibrate_iota(obs["pi0"], obs["X0"], obs["wL0"], p))

    # ---------- 4. DHA 基线 ----------
    base = solve_dha_baseline(obs, p, T)
    base_est = solve_dha_baseline(obs, p_est, T)
    lev = slice(t_obs, None)
    val_rows = [dict(scenario="baseline",
                     max_abs_err_L=np.max(np.abs(base["L"] - truth["L"][lev])),
                     max_abs_err_log_omega=np.max(np.abs(base["log_omega_rel"]
                                                         - (truth["log_omega"][lev] - truth["log_omega"][t_obs]))),
                     max_abs_err_D0=np.nan, welfare_decomposition_gap=np.nan)]
    regions_to_plot = [0, 3, 7]
    plotting.fig_validation(truth["L"][lev], base["L"], p, regions_to_plot,
                            out_fig / "fig1_baseline_validation.png", val_rows[0]["max_abs_err_L"])
    log("DHA 基线完成")

    # ---------- 5. 反事实 ----------
    welfare_rows, summary = [], {}
    for name, sc in make_scenarios(p, geo, T).items():
        dtau = None if sc["dtau_odds"] is None else p.nu * sc["dtau_odds"]
        dtau_est = None if sc["dtau_odds"] is None else p_est.nu * sc["dtau_odds"]
        cf = solve_dha_counterfactual(base, p, Ahat=sc["Ahat"], kaphat=sc["kaphat"], dtau=dtau)
        cf_est = solve_dha_counterfactual(base_est, p_est, Ahat=sc["Ahat"], kaphat=sc["kaphat"], dtau=dtau_est)
        # 已知基本面的水平解（验证用）
        A_cf = hist["A"][lev] * (1.0 if sc["Ahat"] is None else sc["Ahat"])
        kappa_cf = hist["kappa"][lev] * (1.0 if sc["kaphat"] is None else sc["kaphat"])
        tau_cf = None if dtau is None else fund.tau[None] + dtau
        lev_cf = solve_levels_path(truth["L"][t_obs], fund, p, T + 1, A_path=A_cf, kappa_path=kappa_cf,
                                   tau_path=tau_cf)
        w = welfare(cf, base, p)
        D0_lev = lev_cf["V"][0] - truth["V"][t_obs]
        val_rows.append(dict(scenario=name, max_abs_err_L=np.max(np.abs(cf["L"] - lev_cf["L"])),
                             max_abs_err_log_omega=np.max(np.abs(cf["log_omega_hat"]
                                                                 - (lev_cf["log_omega"] - truth["log_omega"][lev]))),
                             max_abs_err_D0=np.max(np.abs(cf["D"][0] - D0_lev)),
                             welfare_decomposition_gap=w["decomposition_gap"]))
        ce_true, nat_true = region_welfare(cf["D"][0], obs["L0"], p)
        ce_est, nat_est = region_welfare(cf_est["D"][0], obs["L0"], p_est)
        ce_lev, _ = region_welfare(D0_lev, obs["L0"], p)
        for n in range(p.n_dom):
            welfare_rows.append(dict(scenario=name, region=p.region_names[n], ce_levels=ce_lev[n],
                                     ce_dha_true_params=ce_true[n], ce_dha_estimated_params=ce_est[n]))
        welfare_rows.append(dict(scenario=name, region="全国", ce_levels=np.nan,
                                 ce_dha_true_params=nat_true, ce_dha_estimated_params=nat_est))
        summary[name] = scenario_figure(name, sc["title"], cf, cf_est, base, base_est, p, geo,
                                        ce_true, ce_est, out_fig / f"fig3_{name}.png")
        summary[name].update(nat_true=nat_true, nat_est=nat_est)
        log(f"反事实 {name} 完成：DHA 与水平解最大误差 {val_rows[-1]['max_abs_err_L']:.1e}")

    validation = pd.DataFrame(val_rows)
    validation.to_csv(out_tab / "validation.csv", index=False)
    welfare_df = pd.DataFrame(welfare_rows)
    welfare_df.to_csv(out_tab / "welfare_by_region.csv", index=False)
    write_results_md(ROOT / "output" / "results.md", p, moments, est_summary, validation, welfare_df,
                     summary, T, R, p_est)
    log("全部完成，结果见 output/")


# --------------------------------------------------------------------------------------
# 汇总与作图的辅助函数
# --------------------------------------------------------------------------------------
def data_moments(truth, obs, p, t_obs):
    mu = obs["mu_prev"]
    L = lm_reshape(obs["L0"], p)
    reg = np.repeat(np.arange(p.n_dom), p.J + 1)
    cross = reg[:, None] != reg[None]
    manuf = p.sector_names.index("制造业")
    imp = obs["pi0"][:p.n_dom, p.n_dom:, manuf].sum(axis=1)
    return dict(stay=float((np.diag(mu) * obs["L0"]).sum() / obs["L0"].sum()),
                cross=float(((mu * cross).sum(axis=1) * obs["L0"]).sum() / obs["L0"].sum()),
                nonemp=float(L[:, 0].sum() / L.sum()),
                import_min=float(imp.min()), import_max=float(imp.max()))


def summarize_mc(mc, p):
    rows = [dict(parameter="ν（OLS，固定 β）", truth=p.nu, mean=mc["nu_ols"].mean(), sd=mc["nu_ols"].std()),
            dict(parameter="ν（2SLS，固定 β）", truth=p.nu, mean=mc["nu_iv"].mean(), sd=mc["nu_iv"].std(),
                 mean_se=mc["se_nu_iv"].mean()),
            dict(parameter="β（2SLS，自由估计）", truth=p.beta, mean=mc["beta_free"].mean(),
                 sd=mc["beta_free"].std())]
    for j, name in enumerate(p.sector_names):
        rows.append(dict(parameter=f"θ（{name}）", truth=p.theta[j], mean=mc[f"theta_{j}"].mean(),
                         sd=mc[f"theta_{j}"].std(), mean_se=mc[f"se_theta_{j}"].mean()))
    df = pd.DataFrame(rows)
    df["bias_pct"] = 100 * (df["mean"] / df["truth"] - 1)
    return df


def scenario_figure(name, title, cf, cf_est, base, base_est, p, geo, ce_true, ce_est, fname):
    T1 = cf["L"].shape[0]
    horizon = min(T1, 161)                      # 画前 40 年
    t_years = np.arange(horizon) / 4
    Lb = lm_reshape(base["L"], p)
    Lc = lm_reshape(cf["L"], p)
    coast = geo["coast"]
    manuf = p.sector_names.index("制造业") + 1
    if name == "foreign_shock":
        vb, vc = Lb[..., manuf], Lc[..., manuf]
        path_ylabel, lr_ylabel = "制造业就业变化（%）", "长期制造业就业变化（%）"
        what = "制造业就业"
    else:
        vb, vc = Lb.sum(axis=2), Lc.sum(axis=2)
        path_ylabel, lr_ylabel = "人口变化（%）", "长期人口变化（%）"
        what = "人口"
    paths = [100 * (vc[:horizon, coast].sum(1) / vb[:horizon, coast].sum(1) - 1),
             100 * (vc[:horizon, ~coast].sum(1) / vb[:horizon, ~coast].sum(1) - 1)]
    lr = 100 * (vc[-1] / vb[-1] - 1)
    labels = p.region_names[:p.n_dom]
    plotting.fig_scenario(title, t_years, paths, ["沿海（R1–R3）", "内陆（R4–R8）"], path_ylabel,
                          labels, lr, lr_ylabel, labels, [ce_true, ce_est], ["真实参数", "估计参数"], fname)
    return dict(what=what, coast_lr=float(paths[0][-1]), inland_lr=float(paths[1][-1]),
                lr_by_region=lr, horizon_years=float(t_years[-1]))


def _md_table(df, floatfmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else format(v, floatfmt.get(c, ".3f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_md(fname, p, m, est, val, wel, summary, T, R, p_est):
    lines = ["# 结果摘要（由 run_pipeline.py 自动生成）", "",
             f"时域 T = {T} 季度；蒙特卡洛 {R} 次。所有数字可通过重新运行脚本复现。", "",
             "## 1. 模拟经济的数据特征（t_obs 期）", "",
             f"- 季度留在原劳动力市场的概率（人口加权）：{m['stay']:.3f}",
             f"- 季度跨地区迁移概率：{100 * m['cross']:.2f}%",
             f"- 非就业人口占比：{100 * m['nonemp']:.1f}%",
             f"- 各地区制造业从国外进口的份额：{100 * m['import_min']:.1f}% – {100 * m['import_max']:.1f}%", "",
             "## 2. 弹性估计（蒙特卡洛）", "",
             _md_table(est.rename(columns=dict(parameter="参数", truth="真值", mean="均值", sd="标准差",
                                               mean_se="平均标准误", bias_pct="偏误（%）")),
                       {"偏误（%）": ".1f"}), "",
             f"下文“估计参数”一列使用第 1 次重复（即研究者手中的一份数据）的估计值：ν̂ = {p_est.nu:.3f}，"
             f"θ̂ = ({', '.join(f'{x:.2f}' for x in p_est.theta)})。这次抽样的 ν̂ 比蒙特卡洛均值高 "
             f"{(p_est.nu - est.loc[1, 'mean']) / est.loc[1, 'sd']:.1f} 个标准差。", "",
             "注：PPML 三向固定效应（进口地×年、出口地×年、地区对）的聚类标准误低于蒙特卡洛标准差，"
             "这与 Weidner & Zylkin (2021, JIE) 的结论一致：点估计一致，但渐近标准误偏小。"
             "真实数据中建议用自助法或偏误校正的方差。", "",
             "## 3. DHA 与已知基本面的水平解对照（最大绝对误差）", "",
             _md_table(val.rename(columns=dict(scenario="情景", max_abs_err_L="劳动力 L",
                                               max_abs_err_log_omega="实际工资 log ω",
                                               max_abs_err_D0="终生价值 D0",
                                               welfare_decomposition_gap="福利分解式误差")),
                       {k: ".1e" for k in ["劳动力 L", "实际工资 log ω", "终生价值 D0", "福利分解式误差"]}), "",
             "## 4. 反事实结果", ""]
    names = dict(foreign_shock="外国制造业生产率冲击", corridor="沿海—内陆交通走廊", migration_reform="跨地区迁移成本下降")
    for key, s in summary.items():
        lines += [f"### {names[key]}", "",
                  f"- {s['horizon_years']:.0f} 年后{s['what']}变化：沿海 {s['coast_lr']:+.2f}%，内陆 {s['inland_lr']:+.2f}%",
                  f"- 全国消费等价福利变化：{s['nat_true']:+.3f}%（真实参数） / {s['nat_est']:+.3f}%（估计参数）", "",
                  _md_table(wel[wel.scenario == key].drop(columns="scenario").rename(
                      columns=dict(region="地区", ce_levels="水平解", ce_dha_true_params="DHA·真实参数",
                                   ce_dha_estimated_params="DHA·估计参数")),
                            {c: "+.3f" for c in ["水平解", "DHA·真实参数", "DHA·估计参数"]}), ""]
    Path(fname).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="快速版：较短时域与较少蒙特卡洛次数")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    main(quick=args.quick, seed=args.seed)
