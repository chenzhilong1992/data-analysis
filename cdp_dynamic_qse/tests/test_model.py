"""正确性测试：用已知基本面的水平解检验精确/动态帽子代数，并检验估计量能恢复真实参数。

运行：在 cdp_dynamic_qse/ 目录下执行 `python -m pytest -q`。
为了速度，这里用一个较小的经济（3 个国内地区 + 1 个外国、2 个部门、年度 β）。
"""
import numpy as np
import pytest

from cdp.dynamics import (solve_dha_baseline, solve_dha_counterfactual, solve_levels_path,
                          stationary_value, welfare)
from cdp.estimation import iv_2sls, ppml
from cdp.model import Fundamentals, Params, employment
from cdp.simulate import observed_cross_section
from cdp.static_eq import calibrate_iota, solve_temp_eq_hat, solve_temp_eq_levels


def small_economy(seed=0):
    rng = np.random.default_rng(seed)
    n_dom, n_for, J = 3, 1, 2
    N = n_dom + n_for
    M = n_dom * (J + 1)
    gamma_va = rng.uniform(0.35, 0.6, (N, J))
    gamma_io = rng.dirichlet(np.ones(J) * 4, (N, J)) * (1 - gamma_va)[..., None]
    p = Params(beta=0.95, nu=2.0, theta=np.array([6.0, 4.0]),
               alpha=rng.dirichlet(np.ones(J) * 5, N), gamma_va=gamma_va, gamma_io=gamma_io,
               xi=rng.uniform(0.15, 0.25, N), iota=rng.dirichlet(np.ones(N) * 3), n_dom=n_dom,
               region_names=["A", "B", "C", "F"], sector_names=["s1", "s2"])
    p.check()
    kappa = np.exp(np.abs(rng.normal(0.4, 0.3, (N, N, J))))
    for j in range(J):
        np.fill_diagonal(kappa[:, :, j], 1.0)
    tau = rng.uniform(2.0, 4.0, (M, M))
    np.fill_diagonal(tau, 0.0)
    fund = Fundamentals(A=np.exp(rng.normal(0, 0.2, (N, J))), kappa=kappa,
                        H=rng.uniform(0.2, 0.4, (N, J)), tau=tau, b=np.full(n_dom, 0.5),
                        L_for=np.array([[0.6, 0.6]]))
    L0 = rng.dirichlet(np.ones(M) * 5)
    return p, fund, L0


@pytest.fixture(scope="module")
def economy():
    p, fund, L0 = small_economy()
    t_obs, T = 6, 120
    rng = np.random.default_rng(1)
    A_path = np.repeat(fund.A[None], t_obs + T + 1, axis=0)
    A_path[1:t_obs + 1] *= np.exp(rng.normal(0, 0.05, (t_obs, p.N, p.J)))
    A_path[t_obs + 1:] = A_path[t_obs]
    path = solve_levels_path(L0, fund, p, t_obs + T + 1, A_path=A_path)
    obs = observed_cross_section(path, fund, t_obs)
    base = solve_dha_baseline(obs, p, T)
    return dict(p=p, fund=fund, path=path, obs=obs, base=base, t_obs=t_obs, T=T, A_path=A_path)


def test_exact_hat_matches_levels():
    p, fund, L0 = small_economy()
    rng = np.random.default_rng(2)
    L = employment(L0, fund.L_for, p)[None]
    L1 = L * rng.uniform(0.8, 1.2, L.shape)
    A1 = fund.A * np.exp(rng.normal(0, 0.1, fund.A.shape))
    k1 = fund.kappa * np.exp(rng.normal(0, 0.1, fund.kappa.shape))
    e0 = solve_temp_eq_levels(L, fund.A, fund.kappa, fund.H, p)
    e1 = solve_temp_eq_levels(L1, A1, k1, fund.H, p)
    h = solve_temp_eq_hat(e0["pi"], e0["wL"], L1 / L, p, Ahat=(A1 / fund.A)[None],
                          kaphat=(k1 / fund.kappa)[None])
    assert np.allclose(h["log_omega_hat"], e1["log_omega"] - e0["log_omega"], atol=1e-10)
    assert np.allclose(h["pi"], e1["pi"], atol=1e-10)
    # ι 可以从观测数据中反推
    assert np.allclose(calibrate_iota(e0["pi"][0], e0["X"][0], e0["wL"][0], p), p.iota, atol=1e-10)


def test_stationary_value_is_fixed_point():
    p, fund, _ = small_economy()
    flow = np.random.default_rng(3).normal(size=p.M)
    V = stationary_value(flow, fund.tau, p.beta, p.nu)
    z = (p.beta * V[None] - fund.tau) / p.nu
    rhs = flow + p.nu * np.log(np.exp(z).sum(axis=1))
    assert np.allclose(V, rhs, atol=1e-10)


def test_dha_baseline_replicates_levels(economy):
    e = economy
    lev = slice(e["t_obs"], None)
    assert np.max(np.abs(e["base"]["L"] - e["path"]["L"][lev])) < 1e-8
    lo = e["path"]["log_omega"][lev] - e["path"]["log_omega"][e["t_obs"]]
    assert np.max(np.abs(e["base"]["log_omega_rel"] - lo)) < 1e-8


@pytest.mark.parametrize("shock", ["productivity", "trade_cost", "migration_cost"])
def test_counterfactual_replicates_levels(economy, shock):
    e = economy
    p, fund, T, t_obs = e["p"], e["fund"], e["T"], e["t_obs"]
    Ahat = kaphat = dtau = None
    A_cf, kappa_cf, tau_cf = e["A_path"][t_obs:], None, None
    if shock == "productivity":
        Ahat = np.ones((T + 1, p.N, p.J))
        Ahat[1:, p.n_dom, 0] = 1.3                   # 外国部门 1 生产率从 t=1 起上升 30%
        A_cf = A_cf * Ahat
    elif shock == "trade_cost":
        kaphat = np.ones((T + 1, p.N, p.N, p.J))
        kaphat[1:, 0, 1] = kaphat[1:, 1, 0] = 0.8    # A–B 之间贸易成本下降 20%
        kappa_cf = fund.kappa[None] * kaphat
    else:
        dtau = np.zeros((T + 1, p.M, p.M))
        reg = np.repeat(np.arange(p.n_dom), p.J + 1)
        dtau[:] = np.where(reg[:, None] != reg[None], -0.5, 0.0)   # 跨地区迁移成本下降
        tau_cf = fund.tau[None] + dtau
    cf = solve_dha_counterfactual(e["base"], p, Ahat=Ahat, kaphat=kaphat, dtau=dtau)
    lev = solve_levels_path(e["path"]["L"][t_obs], fund, p, T + 1, A_path=A_cf,
                            kappa_path=kappa_cf, tau_path=tau_cf)
    assert np.max(np.abs(cf["L"] - lev["L"])) < 1e-8
    D0_levels = lev["V"][0] - e["path"]["V"][t_obs]
    assert np.max(np.abs(cf["D"][0] - D0_levels)) < 1e-7
    assert welfare(cf, e["base"], p)["decomposition_gap"] < 1e-8


def test_ppml_recovers_coefficients():
    rng = np.random.default_rng(4)
    n_g, n = 30, 3000
    g1, g2 = rng.integers(0, n_g, n), rng.integers(0, n_g, n)
    X = rng.normal(size=(n, 2))
    fe = rng.normal(size=n_g)
    mu = np.exp(X @ np.array([-2.0, 0.5]) + fe[g1] + fe[g2])
    y = mu * rng.lognormal(-0.02, 0.2, n)              # 均值为 1 的乘性误差：PPML 仍一致
    coef, se = ppml(y, X, [g1, g2])
    assert np.all(np.abs(coef - [-2.0, 0.5]) < 3 * se)


def test_2sls_corrects_measurement_error():
    rng = np.random.default_rng(5)
    G, T = 400, 30
    g = np.repeat(np.arange(G), T)
    x_true = np.cumsum(rng.normal(size=(G, T)), axis=1).ravel() * 0.3
    y = 1.5 * x_true + rng.normal(size=G)[g] + rng.normal(0, 0.1, G * T)
    x_obs = x_true + rng.normal(0, 0.5, G * T)
    z = np.roll(x_true.reshape(G, T), 1, axis=1).ravel() + rng.normal(0, 0.5, G * T)
    keep = np.tile(np.arange(T) > 0, G)
    ols, _ = iv_2sls(y[keep], x_obs[keep], x_obs[keep], g[keep])
    iv, _ = iv_2sls(y[keep], x_obs[keep], z[keep], g[keep])
    assert ols[0] < 1.4 and abs(iv[0] - 1.5) < 0.05
