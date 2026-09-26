"""构造模拟经济（“真实”数据生成过程）并生成研究者可观测的数据。

设计：8 个国内地区（R1–R3 为沿海，R4–R8 为内陆）+ 2 个外国地区（F1 = 外国 A，F2 = 世界其他），
3 个部门（农业、制造业、服务业），季度频率。国内劳动力可在 地区×(非就业+3 部门) 共 32 个劳动力
市场之间迁移，外国劳动力固定。历史期（t = 0..t_obs）生产率受 AR(1) 冲击，t_obs 之后基本面不变；
研究者在 t_obs 期观测到一张截面（π, X, wL, L, μ_{t_obs-1}），并拥有带抽样误差的历史面板。
"""
import numpy as np

from .model import Fundamentals, Params, employment, lm_reshape
from .static_eq import solve_temp_eq_levels

SECTORS = ["农业", "制造业", "服务业"]


def build_economy(seed=7, n_coast=3, n_inland=5, n_for=2):
    """返回 (Params, Fundamentals, geo)。geo 中保存距离、运费等“可观测”的地理信息。"""
    rng = np.random.default_rng(seed)
    n_dom = n_coast + n_inland
    N, J = n_dom + n_for, len(SECTORS)
    region_names = [f"R{n + 1}" for n in range(n_dom)] + ["F1", "F2"][:n_for]

    # ---------- 份额参数（可从投入产出表直接读出） ----------
    gamma_va = np.clip(np.array([0.45, 0.35, 0.60]) + rng.normal(0, 0.03, (N, J)), 0.2, 0.8)
    io_weights = np.array([[0.30, 0.40, 0.30],     # 农业使用的中间品结构
                           [0.10, 0.60, 0.30],     # 制造业
                           [0.05, 0.30, 0.65]])    # 服务业
    gamma_io = np.stack([np.stack([rng.dirichlet(80 * io_weights[j]) for j in range(J)])
                         for _ in range(N)]) * (1.0 - gamma_va)[..., None]
    alpha = np.stack([rng.dirichlet(200 * np.array([0.10, 0.30, 0.60])) for _ in range(N)])
    xi = 0.2 + rng.uniform(-0.03, 0.03, N)

    # ---------- 地理：坐标、距离、边境 ----------
    x = np.r_[rng.uniform(0.0, 0.25, n_coast), rng.uniform(0.45, 1.0, n_inland)]
    yv = rng.uniform(0.0, 1.0, n_dom)
    dist = np.zeros((N, N))
    dist[:n_dom, :n_dom] = np.hypot(x[:, None] - x[None], yv[:, None] - yv[None])
    for f in range(n_for):
        d_f = 0.4 + 0.4 * f + x + 0.1 * rng.uniform(size=n_dom)       # 沿海离国外更近（港口）
        dist[n_dom + f, :n_dom] = dist[:n_dom, n_dom + f] = d_f
    for f in range(n_for):
        for g in range(n_for):
            if f != g:
                dist[n_dom + f, n_dom + g] = 1.5
    is_for = np.arange(N) >= n_dom
    border = (is_for[:, None] | is_for[None]) & ~np.eye(N, dtype=bool)

    # ---------- 贸易成本：可观测运费 f + 距离 + 边境 + 不可观测成分 η ----------
    theta = np.array([8.0, 5.0, 4.0])
    delta = np.array([0.8, 0.6, 1.5])          # 距离弹性（服务业最难贸易）
    border_cost = np.array([0.15, 0.05, 0.6])
    offdiag = ~np.eye(N, dtype=bool)[..., None]
    freight = np.where(offdiag, np.clip(np.exp(rng.normal(np.log(0.12), 0.6, (N, N, J))), 0, 1.0), 0.0)
    eta = np.where(offdiag, rng.normal(0, 0.10, (N, N, J)), 0.0)
    log_kappa_other = np.where(offdiag, delta * np.log1p(dist)[..., None]
                               + border_cost * border[..., None] + eta, 0.0)
    kappa = np.exp(np.log1p(freight) + log_kappa_other)

    # ---------- 生产率、结构、外国劳动力 ----------
    A = np.exp(rng.normal(0, 0.15, (N, J)))
    L_for = np.array([[0.45, 0.60, 0.45], [0.9, 1.2, 0.9]])[:n_for]   # 外国就业（国内总人口 = 1）
    sector_mix = np.array([0.25, 0.30, 0.45])
    H = np.exp(rng.normal(0, 0.2, (N, J))) * np.r_[np.full(n_dom, 1.0 / n_dom), np.ones(n_for) * 1.5][:, None] * sector_mix

    # ---------- 迁移成本（效用单位 τ = ν·c） ----------
    nu, beta = 5.34, 0.99                       # CDP (2019) 的季度值
    M = n_dom * (J + 1)
    reg = np.repeat(np.arange(n_dom), J + 1)
    sec = np.tile(np.arange(J + 1), n_dom)
    c = np.zeros((M, M))
    cross = reg[:, None] != reg[None]
    c += cross * (5.5 + 2.0 * dist[reg][:, reg])
    switch = sec[:, None] != sec[None]
    involves_ne = (sec[:, None] == 0) | (sec[None] == 0)
    c += switch * np.where(involves_ne, 3.0, 4.0)
    c += rng.normal(0, 0.2, (M, M))
    np.fill_diagonal(c, 0.0)
    tau = nu * c

    # ---------- 资产组合份额 ι ----------
    size = np.r_[np.full(n_dom, 1.0 / n_dom), L_for.sum(axis=1)]
    iota = size * np.exp(rng.normal(0, 0.25, N))
    iota /= iota.sum()

    p = Params(beta=beta, nu=nu, theta=theta, alpha=alpha, gamma_va=gamma_va, gamma_io=gamma_io,
               xi=xi, iota=iota, n_dom=n_dom, region_names=region_names, sector_names=SECTORS)
    p.check()

    # ---------- 家庭生产 b：设为本地区平均实际工资的一定比例 ----------
    L_guess = np.tile(np.r_[0.08, 0.92 * sector_mix], n_dom) / n_dom
    eq = solve_temp_eq_levels(employment(L_guess, L_for, p)[None], A, kappa, H, p)
    b = np.exp(eq["log_omega"][0, :n_dom].mean(axis=1) - 0.35)

    fund = Fundamentals(A=A, kappa=kappa, H=H, tau=tau, b=b, L_for=L_for)
    geo = dict(dist=dist, border=border, freight=freight, log_kappa_other=log_kappa_other,
               coast=np.arange(n_dom) < n_coast, x=x, y=yv, delta=delta, border_cost=border_cost, eta=eta)
    return p, fund, geo


def initial_distribution(p, seed=11):
    """t=0 的国内劳动力分布：偏离稳态（人口在地区与部门之间分布不均），从而产生转移动态。"""
    rng = np.random.default_rng(seed)
    region = rng.dirichlet(np.full(p.n_dom, 6.0))
    L = np.stack([rng.dirichlet(120 * np.array([0.10, 0.28, 0.27, 0.35])) for _ in range(p.n_dom)])
    return (L * region[:, None]).reshape(-1)


def history_paths(fund, geo, p, t_obs, T_total, seed=3):
    """历史期（t = 1..t_obs）的基本面冲击；t ≥ t_obs 时冻结（此后基本面不变）。

    - 生产率：A_t = A · exp(s_t)，s_t 为 AR(1)（ρ = 0.85，σ = 0.03）；
    - 运费：log f_t = log f_0 + g_t，g_t 为 AR(1)（ρ = 0.95，σ = 0.10），
      κ_t = (1 + f_t) · exp(距离、边境与不可观测成分)。运费的时间变化用于识别贸易弹性。
    完全预见：t=0 时经济主体就知道整条路径。
    """
    rng = np.random.default_rng(seed)
    s = np.zeros((T_total, p.N, p.J))
    g = np.zeros((T_total, p.N, p.N, p.J))
    for t in range(1, t_obs + 1):
        s[t] = 0.85 * s[t - 1] + 0.03 * rng.normal(size=(p.N, p.J))
        g[t] = 0.95 * g[t - 1] + 0.10 * rng.normal(size=(p.N, p.N, p.J))
    s[t_obs + 1:] = s[t_obs]
    g[t_obs + 1:] = g[t_obs]
    freight = geo["freight"][None] * np.exp(g)
    kappa = np.exp(np.log1p(freight) + geo["log_kappa_other"][None])
    return dict(A=fund.A[None] * np.exp(s), kappa=kappa, freight=freight)


def observed_cross_section(path, fund, t):
    """研究者在 t 期观测到的截面：动态帽子代数只需要这些。"""
    return dict(L0=path["L"][t].copy(), mu_prev=path["mu"][t - 1].copy(), pi0=path["pi"][t].copy(),
                X0=path["X"][t].copy(), wL0=path["wL"][t].copy(), L_for=fund.L_for.copy())


def noisy_migration_panel(path, p, t_end, rng, sample_size=4e6, sigma_w=0.03):
    """带抽样误差的历史面板：t = 0..t_end-1 的迁移流（多项分布抽样）与 t = 0..t_end 的实际工资。

    迁移流：每期从各劳动力市场按人口比例抽取样本（总样本量 sample_size），统计去向频率；
    实际工资：log ω 加上 N(0, σ_w²) 的测量误差。非就业没有工资数据（记为 0，由配对固定效应吸收）。
    """
    M = p.M
    mu_obs = np.empty((t_end, M, M))
    for t in range(t_end):
        n_a = np.maximum(np.round(sample_size * path["L"][t] / path["L"][t].sum()), 1).astype(np.int64)
        for a in range(M):
            mu_obs[t, a] = rng.multinomial(n_a[a], path["mu"][t, a]) / n_a[a]
    logw_obs = path["log_omega"][: t_end + 1] + rng.normal(0, sigma_w, (t_end + 1, M))
    ne = lm_reshape(np.arange(M), p)[:, 0]
    logw_obs[:, ne] = 0.0
    return dict(mu=mu_obs, log_omega=logw_obs)


def noisy_trade_panel(path, hist, periods, rng, sigma_x=0.10):
    """双边贸易额 X^{nj,ij}_t = π^{nj,ij}_t X^{nj}_t，乘以均值为 1 的对数正态测量误差；附带各期运费。"""
    periods = np.asarray(periods)
    flows = path["pi"][periods] * path["X"][periods][:, :, None, :]
    noise = np.exp(rng.normal(-0.5 * sigma_x ** 2, sigma_x, flows.shape))
    return dict(flows=flows * noise, periods=periods, freight=hist["freight"][periods])
