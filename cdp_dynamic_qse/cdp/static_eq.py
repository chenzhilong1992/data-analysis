"""临时均衡（给定劳动力配置时的静态贸易均衡）。

两个入口共用同一个求解内核：
- solve_temp_eq_levels：已知基本面水平值（A, κ, H）时的水平解，用于生成“真实”数据；
- solve_temp_eq_hat：精确帽子代数（exact hat algebra），只需参照期的贸易份额 π 与劳动收入 wL，
  以及劳动力与基本面的变化，用于动态帽子代数（DHA）。

所有数组都带一个前导批量维度 B（通常是时间），各期相互独立，可以一次性并行求解。
"""
import numpy as np

from .accel import Anderson


def _lse(z, axis):
    """数值稳定的 log-sum-exp（比 scipy 版本开销小，这里会被调用上万次）。"""
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(z - m), axis=axis))


def _expenditure(pi, income, p):
    """求解商品市场出清的线性方程组，得到各地区各部门的总支出 X（最终 + 中间）。

    X^{nj} = Σ_k γ^{nk,nj} Σ_i π^{ik,nk} X^{ik} + α^{nj} I^n
    pi: (B, N, N, J)，income: (B, N) -> X: (B, N, J)
    """
    B, N, _, J = pi.shape
    # Mmat[b, n, j, i, k] = γ^{nk,nj} · π^{ik,nk}：地区 i 部门 k 的支出中，经由购买 n 的 k 部门产品
    # 而转化为对 n 的 j 部门中间品的需求
    Mmat = np.einsum("nkj,bink->bnjik", p.gamma_io, pi).reshape(B, N * J, N * J)
    rhs = (p.alpha[None] * income[:, :, None]).reshape(B, N * J, 1)
    X = np.linalg.solve(np.eye(N * J)[None] - Mmat, rhs)
    return X.reshape(B, N, J)


def _price_fixed_point(base, zconst, logP, p, tol, maxit=500):
    """在给定工资下求解价格指数的不动点（投入产出联系使价格互相依赖）。

    单位成本 log x^{ij} = base^{ij} + Σ_k γ^{ij,ik} log P^{ik}
    价格指数 log P^{nj} = -(1/θ^j) log Σ_i exp(zconst^{nj,ij} - θ^j log x^{ij})
    """
    for _ in range(maxit):
        logx = base + np.einsum("njk,bnk->bnj", p.gamma_io, logP)
        z = zconst - p.theta * logx[:, None]
        logP_new = -_lse(z, axis=2) / p.theta
        err = np.max(np.abs(logP_new - logP))
        logP = logP_new
        if err < tol:
            break
    logx = base + np.einsum("njk,bnk->bnj", p.gamma_io, logP)
    z = zconst - p.theta * logx[:, None]
    return logP, z


def _solve_core(S, base0, zconst, target, p, logw, logP, tol, maxit, damp):
    """求解内核：在工资上做不动点迭代。

    S:      (B, N, J) 满足 wL = exp(logw) · S（水平解中 S = L；帽子解中 S = L̂ · wL_ref，logw = log ŵ）
    base0:  (B, N, J) 单位成本中除 γ·log w 与中间品之外的部分
    zconst: (B, N, N, J) 贸易份额分子中与工资无关的部分
    target: (B,) 名义计价单位：Σ wL 的目标值
    """
    xi = p.xi[:, None]
    rent_ratio = xi / (1.0 - xi)          # 结构租金 / 劳动收入 = ξ/(1-ξ)
    # 全步长更新 w = γ(1-ξ)Y/L 会超调（劳动需求对自身工资的弹性约为 -θγ），
    # 因此按 1/(1+θγ) 缩放步长，相当于对角近似的牛顿步
    step = damp / (1.0 + p.theta * p.gamma_va)
    B = S.shape[0]
    acc = Anderson(m=6)
    err, best = 1.0, np.inf
    for it in range(maxit):
        base = p.gamma_va * logw + base0
        # 价格内循环的精度随工资误差自适应收紧，避免在远离均衡时白白迭代
        logP, z = _price_fixed_point(base, zconst, logP, p, tol=max(tol * 0.1, err * 1e-2))
        pi = np.exp(z - _lse(z, axis=2)[:, :, None])
        wL = np.exp(logw) * S
        chi = np.sum(wL * rent_ratio, axis=(1, 2))                # 全球结构租金池 χ
        income = wL.sum(axis=2) + p.iota[None] * chi[:, None]     # 地区收入 I^n
        X = _expenditure(pi, income, p)
        Y = np.einsum("bnij,bnj->bij", pi, X)                     # 地区 i 部门 j 的总销售额
        logw_new = np.log(p.gamma_va * (1.0 - xi) * Y / S)
        logw_new -= np.log(np.sum(np.exp(logw_new) * S, axis=(1, 2)) / target)[:, None, None]
        err = np.max(np.abs(logw_new - logw))
        if err < tol:
            break
        g = logw + step * (logw_new - logw)                     # 阻尼后的不动点映射
        if err > 10 * best or not np.isfinite(err):             # 加速失稳时退回普通迭代
            acc.reset()
        best = min(best, err)
        logw = acc.update(logw.reshape(B, -1), g.reshape(B, -1)).reshape(g.shape)
        logw -= np.log(np.sum(np.exp(logw) * S, axis=(1, 2)) / target)[:, None, None]
    else:
        raise RuntimeError(f"临时均衡未收敛：max|Δlog w| = {err:.2e}")
    # 用收敛后的工资重算一次，保证输出彼此一致
    base = p.gamma_va * logw + base0
    logP, z = _price_fixed_point(base, zconst, logP, p, tol=tol * 0.1)
    pi = np.exp(z - _lse(z, axis=2)[:, :, None])
    wL = np.exp(logw) * S
    chi = np.sum(wL * rent_ratio, axis=(1, 2))
    income = wL.sum(axis=2) + p.iota[None] * chi[:, None]
    X = _expenditure(pi, income, p)
    logPc = np.sum(p.alpha[None] * logP, axis=2)                  # 消费者价格指数 log P^n
    return dict(logw=logw, logP=logP, logPc=logPc, pi=pi, X=X, wL=wL, income=income,
                log_omega=logw - logPc[:, :, None], iterations=it + 1)


def solve_temp_eq_levels(L, A, kappa, H, p, logw0=None, logP0=None, tol=1e-12, maxit=5000, damp=1.0):
    """水平值临时均衡。L: (B,N,J)；A: (B,N,J) 或 (N,J)；kappa: (B,N,N,J) 或 (N,N,J)；H: (N,J)。

    计价单位：全球劳动收入 Σ wL = 1。返回的 log_omega = log(w / P^n) 为实际工资。
    """
    B, N, J = L.shape
    xi = p.xi[:, None]
    logA = np.broadcast_to(np.log(A), (B, N, J))
    logk = np.broadcast_to(np.log(kappa), (B, N, N, J))
    # 单位成本 x = [r^ξ w^{1-ξ}]^γ Π_k P_k^{γ_k}，其中 r = ξ/(1-ξ) · wL / H
    base0 = p.gamma_va * (xi * (np.log(L) - np.log(H)) + xi * np.log(xi / (1.0 - xi)))
    zconst = -p.theta * logk + (p.theta * p.gamma_va * logA)[:, None]
    logw = np.zeros((B, N, J)) if logw0 is None else logw0.copy()
    logP = np.zeros((B, N, J)) if logP0 is None else logP0.copy()
    return _solve_core(L, base0, zconst, np.ones(B), p, logw, logP, tol, maxit, damp)


def solve_temp_eq_hat(pi_ref, wL_ref, Lhat, p, Ahat=None, kaphat=None, logw0=None, logP0=None,
                      tol=1e-12, maxit=5000, damp=1.0):
    """精确帽子代数下的临时均衡：相对于参照均衡 (pi_ref, wL_ref) 的变化。

    pi_ref: (B,N,N,J)；wL_ref: (B,N,J)；Lhat: (B,N,J) 就业变化 L'/L；
    Ahat: 生产率变化 A'/A；kaphat: 贸易成本变化 κ'/κ。
    计价单位：Σ wL' = Σ wL_ref。返回 log_omega_hat = log(ŵ / P̂^n) 为实际工资变化。
    """
    B, N, J = Lhat.shape
    xi = p.xi[:, None]
    logAhat = np.zeros((B, N, J)) if Ahat is None else np.broadcast_to(np.log(Ahat), (B, N, J))
    logkhat = 0.0 if kaphat is None else np.broadcast_to(np.log(kaphat), (B, N, N, J))
    with np.errstate(divide="ignore"):
        logpi = np.log(pi_ref)                       # 允许零贸易流（log 0 = -inf）
    base0 = p.gamma_va * xi * np.log(Lhat)           # 结构固定，r̂ = ŵ L̂
    zconst = logpi - p.theta * logkhat + (p.theta * p.gamma_va * logAhat)[:, None]
    logw = np.zeros((B, N, J)) if logw0 is None else logw0.copy()
    logP = np.zeros((B, N, J)) if logP0 is None else logP0.copy()
    target = wL_ref.sum(axis=(1, 2))
    out = _solve_core(Lhat * wL_ref, base0, zconst, target, p, logw, logP, tol, maxit, damp)
    out["log_omega_hat"] = out.pop("log_omega")
    return out


def calibrate_iota(pi, X, wL, p):
    """由数据（π, X, wL 与投入产出份额）反推资产组合份额 ι^n。

    最终支出 F^{nj} = X^{nj} - Σ_k γ^{nk,nj} Y^{nk}，地区收入 I^n = Σ_j F^{nj}，
    ι^n = (I^n - Σ_j wL^{nj}) / χ，χ = Σ ξ/(1-ξ) wL。
    """
    Y = np.einsum("nij,nj->ij", pi, X)
    F = X - np.einsum("nkj,nk->nj", p.gamma_io, Y)
    income = F.sum(axis=1)
    chi = np.sum(wL * (p.xi / (1.0 - p.xi))[:, None])
    return (income - wL.sum(axis=1)) / chi
