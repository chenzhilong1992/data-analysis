"""弹性估计：贸易弹性 θ（PPML 引力方程）与迁移弹性 1/ν（ACM/CDP 欧拉型方程，2SLS）。

这两类弹性加上可直接观测的份额参数，就是动态帽子代数所需的全部参数。
"""
import numpy as np


# --------------------------------------------------------------------------------------
# 通用估计量
# --------------------------------------------------------------------------------------
def _dummies(codes, drop_first):
    levels, inv = np.unique(codes, return_inverse=True)
    D = np.zeros((codes.size, levels.size))
    D[np.arange(codes.size), inv] = 1.0
    return D[:, 1:] if drop_first else D


def ppml(y, X, fe_codes, cluster=None, tol=1e-12, maxit=200):
    """带固定效应的泊松伪极大似然（IRLS）。返回 (系数, 稳健标准误)，只报告 X 的部分。

    y: (n,) 非负；X: (n, k)；fe_codes: 若干个 (n,) 整数编码的固定效应（允许多组固定效应之间
    存在共线性，用最小二乘/伪逆处理，X 的系数不受影响）；cluster: 聚类编码，缺省为异方差稳健。
    """
    y = y / y.mean()                                   # 只影响常数项，改善数值条件
    Z = np.hstack([X] + [_dummies(c, drop_first=(i > 0)) for i, c in enumerate(fe_codes)])
    mu = 0.5 * (y + y.mean())
    eta = np.log(mu)
    for _ in range(maxit):
        z = eta + (y - mu) / mu
        sw = np.sqrt(mu)
        coef = np.linalg.lstsq(Z * sw[:, None], z * sw, rcond=None)[0]
        eta_new = Z @ coef
        mu = np.exp(eta_new)
        if np.max(np.abs(eta_new - eta)) < tol:
            break
        eta = eta_new
    Hinv = np.linalg.pinv(Z.T @ (Z * mu[:, None]), rcond=1e-12, hermitian=True)
    u = Z * (y - mu)[:, None]
    if cluster is not None:
        _, inv = np.unique(cluster, return_inverse=True)
        s = np.zeros((inv.max() + 1, Z.shape[1]))
        np.add.at(s, inv, u)
        u = s
    V = Hinv @ (u.T @ u) @ Hinv
    k = X.shape[1]
    return coef[:k], np.sqrt(np.diag(V)[:k])


def _within(x, groups):
    """按组去均值（固定效应的 within 变换）。x: (n,) 或 (n, k)。"""
    x2 = x.reshape(len(groups), -1)
    _, inv = np.unique(groups, return_inverse=True)
    sums = np.zeros((inv.max() + 1, x2.shape[1]))
    np.add.at(sums, inv, x2)
    counts = np.bincount(inv)[:, None]
    return (x2 - (sums / counts)[inv]).reshape(x.shape)


def iv_2sls(y, X, Z, groups):
    """组内变换后的 2SLS（Z = X 时即 OLS），按组聚类的稳健标准误。"""
    y, X, Z = _within(y, groups), _within(X, groups), _within(Z, groups)
    X = X.reshape(len(y), -1)
    Z = Z.reshape(len(y), -1)
    PzX = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)          # 第一阶段拟合值
    coef = np.linalg.solve(PzX.T @ X, PzX.T @ y)
    e = y - X @ coef
    _, inv = np.unique(groups, return_inverse=True)
    score = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(score, inv, PzX * e[:, None])
    bread = np.linalg.inv(PzX.T @ X)
    V = bread @ (score.T @ score) @ bread.T
    return coef, np.sqrt(np.diag(V))


def first_stage_F(x, Z, groups):
    """单个内生变量的第一阶段聚类稳健 Wald/F 统计量。"""
    x, Z = _within(x, groups), _within(Z, groups).reshape(len(x), -1)
    pi = np.linalg.solve(Z.T @ Z, Z.T @ x)
    u = x - Z @ pi
    _, inv = np.unique(groups, return_inverse=True)
    score = np.zeros((inv.max() + 1, Z.shape[1]))
    np.add.at(score, inv, Z * u[:, None])
    bread = np.linalg.inv(Z.T @ Z)
    V = bread @ (score.T @ score) @ bread
    return float(pi @ np.linalg.solve(V, pi) / Z.shape[1])


# --------------------------------------------------------------------------------------
# 贸易弹性：PPML 引力方程
# --------------------------------------------------------------------------------------
def estimate_trade_elasticity(trade):
    """逐部门估计面板引力方程（PPML）：

      X^{nj,ij}_t = exp(FE^{nj}_t + FE^{ij}_t + FE^{ni,j}) · (1 + f^{nj,ij}_t)^{-θ^j}

    进口地×年份、出口地×年份固定效应吸收多边阻力与规模，地区对固定效应吸收距离、边境与
    不可观测的贸易成本 η；θ 由地区对内部运费的时间变化识别。标准误按地区对聚类。

    trade: dict(flows (P,N,N,J), freight (P,N,N,J))
    """
    flows, freight = trade["flows"], trade["freight"]
    P, N, _, J = flows.shape
    t_idx, n_idx, i_idx = (a.ravel() for a in np.meshgrid(np.arange(P), np.arange(N), np.arange(N),
                                                            indexing="ij"))
    imp_fe, exp_fe, pair = t_idx * N + n_idx, t_idx * N + i_idx, n_idx * N + i_idx
    off = n_idx != i_idx                     # 本地流量的运费恒为 0，不提供识别信息，但保留以估计多边阻力
    out = []
    for j in range(J):
        y = flows[..., j].ravel()
        X = np.log1p(freight[..., j].ravel())[:, None]
        coef, se = ppml(y, X, [imp_fe, exp_fe, pair], cluster=pair)
        out.append(dict(theta=-coef[0], se=se[0], n_obs=y.size, n_pairs=int(off.sum() / P)))
    return out


# --------------------------------------------------------------------------------------
# 迁移弹性：ACM (2010) / CDP (2019) 欧拉型方程
# --------------------------------------------------------------------------------------
def migration_regression_data(panel, min_flow=0.0):
    """构造回归样本。对 a≠b、1 ≤ t ≤ T-2：

      y_t   = log(μ^{ab}_t / μ^{aa}_t)
      x1    = log ω^b_{t+1} - log ω^a_{t+1}            （实际工资差）
      x2    = log(μ^{ab}_{t+1} / μ^{bb}_{t+1})
      z1,z2 = 上面两项在 t-1 期的取值（工具变量）
    模型含义：y_t = C^{ab} + (β/ν)·x1 + β·x2 + 误差。
    min_flow：只保留样本期平均迁移份额不低于该值的配对。小流量的抽样频率取对数后有 Jensen 偏误，
    且零流量观测只能剔除（按结果选样）；按配对平均值筛选只依赖配对层面的信息，由配对固定效应吸收。
    剩余的零流量观测被剔除。
    """
    mu, lw = panel["mu"], panel["log_omega"]
    Tm, M, _ = mu.shape
    keep = ~np.eye(M, dtype=bool) & (mu.mean(axis=0) >= min_flow)
    a, b = np.nonzero(keep)
    rows = []
    with np.errstate(divide="ignore"):
        lmu = np.log(mu)
    own = np.diagonal(lmu, axis1=1, axis2=2)          # (Tm, M)
    for t in range(1, Tm - 1):
        y = lmu[t, a, b] - own[t, a]
        x1 = lw[t + 1, b] - lw[t + 1, a]
        x2 = lmu[t + 1, a, b] - own[t + 1, b]
        z1 = lw[t - 1, b] - lw[t - 1, a]
        z2 = lmu[t - 1, a, b] - own[t - 1, b]
        pair = a * M + b
        rows.append(np.column_stack([y, x1, x2, z1, z2, pair, np.full(a.size, t)]))
    d = np.vstack(rows)
    d = d[np.all(np.isfinite(d), axis=1)]
    return dict(y=d[:, 0], x1=d[:, 1], x2=d[:, 2], z1=d[:, 3], z2=d[:, 4],
                pair=d[:, 5].astype(np.int64), t=d[:, 6].astype(np.int64))


def estimate_migration_elasticity(panel, beta, min_flow=0.0):
    """返回三种估计：
    - ols_fixed_beta：固定 β，OLS（工资测量误差 → 衰减偏误 → ν 被高估）；
    - iv_fixed_beta：固定 β，用 t-1 期工资差作工具变量的 2SLS（CDP 的做法）；
    - iv_free_beta：同时估计 β 与 β/ν，工具变量为 t-1 期的工资差与迁移比。
    """
    d = migration_regression_data(panel, min_flow)
    g = d["pair"]
    res = {}
    y_tilde = d["y"] - beta * d["x2"]
    for name, Z in [("ols_fixed_beta", d["x1"]), ("iv_fixed_beta", d["z1"])]:
        coef, se = iv_2sls(y_tilde, d["x1"], Z, g)
        nu = beta / coef[0]
        res[name] = dict(nu=nu, se_nu=beta * se[0] / coef[0] ** 2, coef=coef[0], se=se[0])
    res["iv_fixed_beta"]["first_stage_F"] = first_stage_F(d["x1"], d["z1"], g)
    X = np.column_stack([d["x1"], d["x2"]])
    Z = np.column_stack([d["z1"], d["z2"]])
    coef, se = iv_2sls(d["y"], X, Z, g)
    res["iv_free_beta"] = dict(nu=coef[1] / coef[0], beta=coef[1], se_beta=se[1],
                               coef=coef[0], se=se[0])
    res["n_obs"] = d["y"].size
    res["n_pairs"] = np.unique(g).size
    return res
