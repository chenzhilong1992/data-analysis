"""动态均衡：水平解（数据生成过程）、动态帽子代数基线、以及反事实。

时间约定：t 期初劳动力配置 L_t 给定 → 生产与消费（临时均衡）→ 期末按 V_{t+1} 做迁移决策 μ_t
→ L_{t+1} = μ_t' L_t。所有劳动力市场向量长度为 M = n_dom·(J+1)。
"""
import numpy as np

from .accel import Anderson
from .model import employment, lm_log_omega
from .static_eq import _lse, solve_temp_eq_hat, solve_temp_eq_levels


def _log_softmax_rows(z):
    return z - _lse(z, axis=-1)[..., None]


def stationary_value(flow, cost, beta, nu, V0=None, tol=1e-13, maxit=100):
    """求解稳态价值 V = flow + ν log Σ_b exp((βV_b - cost_ab)/ν)（牛顿法）。

    水平解中 flow = log ω、cost = τ；反事实终点中 flow = log ω̂、cost = -ν log μ + Δτ。
    Jacobian 为 I - βμ，因为 β<1 且 μ 为随机矩阵，它总是可逆的。
    """
    M = flow.shape[0]
    V = flow / (1.0 - beta) if V0 is None else V0.copy()
    for _ in range(maxit):
        z = (beta * V[None, :] - cost) / nu
        F = V - flow - nu * _lse(z, axis=1)
        if np.max(np.abs(F)) < tol:
            return V
        mu = np.exp(_log_softmax_rows(z))
        V = V - np.linalg.solve(np.eye(M) - beta * mu, F)
    raise RuntimeError("稳态价值函数未收敛")


def _forward_labor(L0, mu):
    """L_{t+1} = μ_t' L_t。mu: (T, M, M)，返回 (T+1, M)。"""
    L = np.empty((mu.shape[0] + 1, L0.shape[0]))
    L[0] = L0
    for t in range(mu.shape[0]):
        L[t + 1] = L[t] @ mu[t]
    return L


def _broadcast_path(x, shape):
    return np.broadcast_to(x, shape)


class _OuterIteration:
    """外层不动点迭代的更新规则：先做阻尼迭代，误差足够小后切换到 Anderson 加速。"""

    def __init__(self, damp, switch=np.inf):
        self.damp, self.switch = damp, switch
        self.acc = Anderson(m=8)
        self.best = np.inf

    def __call__(self, x, x_new, err):
        g = x + self.damp * (x_new - x)
        if err > self.switch:
            return g
        if err > 10 * self.best:                   # 加速失稳时清空历史
            self.acc.reset()
        self.best = min(self.best, err)
        return self.acc.update(x.reshape(1, -1), g.reshape(1, -1)).reshape(x.shape)


# --------------------------------------------------------------------------------------
# 1. 水平解：已知全部基本面，完全预见。用于生成模拟数据与验证 DHA。
# --------------------------------------------------------------------------------------
def solve_levels_path(L0, fund, p, T, A_path=None, kappa_path=None, tau_path=None,
                      tol=1e-10, maxit=3000, damp=0.5, verbose=False):
    """给定 t=0 的国内劳动力配置 L0 (M,) 与基本面路径，求完全预见的序贯均衡（共 T 期）。

    A_path: (T,N,J)，kappa_path: (T,N,N,J)，tau_path: (T,M,M)（第 t 期迁移决策面对的成本）；
    缺省时取 fund 中的常数值。第 T-1 期假设经济已处于稳态（终端条件）。
    返回各期 L、μ、V、log ω 以及临时均衡（π, X, wL, w, P）。
    """
    N, J, M = p.N, p.J, p.M
    A = _broadcast_path(fund.A if A_path is None else A_path, (T, N, J))
    kappa = _broadcast_path(fund.kappa if kappa_path is None else kappa_path, (T, N, N, J))
    tau = _broadcast_path(fund.tau if tau_path is None else tau_path, (T, M, M))
    log_b = np.log(fund.b)

    # 与 CDP 的算法一样在价值函数路径上迭代：猜 V → μ → L → 临时均衡 → 新 V。
    # 价值不受正数约束，可以从一开始就用 Anderson 加速（在 L 上迭代容易振荡）。
    V = None
    logw = logP = None
    V_end = None
    err = err_eq = 1e-2
    step = _OuterIteration(damp)
    for it in range(maxit):
        # (a) 迁移份额与劳动力运动方程（第一轮用不变的初始分布）
        if V is None:
            L = np.tile(L0, (T, 1))
        else:
            V_next = np.concatenate([V[1:], V[T - 1:]], axis=0)
            logmu = _log_softmax_rows((p.beta * V_next[:, None, :] - tau) / p.nu)
            L = _forward_labor(L0, np.exp(logmu[:-1]))
        # (b) 各期临时均衡（并行）：精度随外层误差自适应收紧
        eq = solve_temp_eq_levels(employment(L, fund.L_for, p), A, kappa, fund.H, p,
                                  logw0=logw, logP0=logP, tol=max(tol * 1e-2, err_eq * 1e-4))
        logw, logP = eq["logw"], eq["logP"]
        log_omega = lm_log_omega(eq["log_omega"], log_b, p)          # (T, M)
        # (c) 价值函数：终点为稳态，其余向后递推
        V_new = np.empty((T, M))
        V_new[T - 1] = V_end = stationary_value(log_omega[T - 1], tau[T - 1], p.beta, p.nu, V0=V_end)
        for t in range(T - 2, -1, -1):
            V_new[t] = log_omega[t] + p.nu * _lse((p.beta * V_new[t + 1][None, :] - tau[t]) / p.nu, axis=1)
        if V is None:
            V = V_new
            continue
        err = np.max(np.abs(V_new - V))
        err_eq = err
        if verbose and it % 20 == 0:
            print(f"  [levels] iter {it:4d}  max|ΔV| = {err:.2e}")
        if err < tol:
            break
        V = step(V, V_new, err)
    else:
        raise RuntimeError(f"水平解未收敛：max|ΔV| = {err:.2e}")
    # 输出最后一轮彼此一致的 (μ, L, 临时均衡)，以及由其得到的 V（与上一轮之差 < tol）
    return dict(L=L, mu=np.exp(logmu), logmu=logmu, V=V_new, log_omega=log_omega, pi=eq["pi"], X=eq["X"],
                wL=eq["wL"], logw=eq["logw"], logP=eq["logP"], logPc=eq["logPc"], iterations=it + 1)


# --------------------------------------------------------------------------------------
# 2. 动态帽子代数：基线（CDP 命题 2）。只需观测数据 + 弹性，不需要基本面水平值。
# --------------------------------------------------------------------------------------
def solve_dha_baseline(obs, p, T, tol=1e-11, maxit=3000, damp=0.5, verbose=False):
    """从 t=0 的观测数据出发，求基本面不变时的基线序贯均衡（时间差分形式）。

    obs 需要：L0 (M,) 国内劳动力配置；mu_prev (M,M) 上一期（t=-1→0）的迁移份额；
             pi0 (N,N,J) 贸易份额；wL0 (N,J) 劳动收入；L_for (n_for,J) 外国就业。
    未知量为 y_t = log u̇_t = V_t - V_{t-1}（t=1..T），终端 y_{T+1} = 0（收敛到稳态）。
      μ_t   = μ_{t-1} · exp(β/ν · y_{t+1}) / Σ(·)                        (t ≥ 0)
      y_t   = log ω̇_t + ν log Σ_k μ_{t-1}^{·k} exp(β/ν · y_{t+1}^k)     (t ≥ 1)
      L_{t+1} = μ_t' L_t，ω̇_t 由临时均衡的精确帽子代数给出。
    实现上各期临时均衡都相对 t=0 的观测均衡求解（与逐期链式求解代数等价），因此可以并行。
    """
    M = p.M
    bn = p.beta / p.nu
    logmu_prev = np.log(obs["mu_prev"])
    Lp0 = employment(obs["L0"], obs["L_for"], p)
    pi_ref = np.broadcast_to(obs["pi0"], (T + 1,) + obs["pi0"].shape)
    wL_ref = np.broadcast_to(obs["wL0"], (T + 1,) + obs["wL0"].shape)

    y = np.zeros((T + 2, M))
    logw = logP = None
    err_eq = 1e-4
    step = _OuterIteration(damp)
    for it in range(maxit):
        # (a) 给定 y，前向得到迁移份额与劳动力
        logmu = np.empty((T + 1, M, M))
        prev = logmu_prev
        for t in range(T + 1):
            prev = logmu[t] = _log_softmax_rows(prev + bn * y[t + 1][None, :])
        mu = np.exp(logmu)
        L = _forward_labor(obs["L0"], mu[:-1])
        # (b) 临时均衡（相对 t=0）：实际工资的累计变化 log(ω_t/ω_0)
        Lhat = employment(L, obs["L_for"], p) / Lp0
        eq = solve_temp_eq_hat(pi_ref, wL_ref, Lhat, p, logw0=logw, logP0=logP,
                               tol=max(tol * 0.1, err_eq * 1e-3))
        logw, logP = eq["logw"], eq["logP"]
        log_omega_rel = lm_log_omega(eq["log_omega_hat"], 0.0, p)      # (T+1, M)，非就业为 0
        dlog_omega = np.diff(log_omega_rel, axis=0)                       # log ω̇_t，t=1..T
        # (c) 向后更新 y
        y_new = np.zeros_like(y)
        for t in range(T, 0, -1):
            y_new[t] = dlog_omega[t - 1] + p.nu * _lse(logmu[t - 1] + bn * y_new[t + 1][None, :], axis=1)
        err = np.max(np.abs(y_new - y))
        err_eq = err
        if verbose and it % 20 == 0:
            print(f"  [DHA baseline] iter {it:4d}  max|Δy| = {err:.2e}")
        if err < tol:
            break
        y = step(y, y_new, err)
    else:
        raise RuntimeError(f"DHA 基线未收敛：max|Δy| = {err:.2e}")
    return dict(L=L, mu=mu, logmu=logmu, y=y, log_omega_rel=log_omega_rel, pi=eq["pi"],
                X=eq["X"], wL=eq["wL"], L_for=obs["L_for"], iterations=it + 1)


# --------------------------------------------------------------------------------------
# 3. 反事实：相对基线的水平差 D_t = V'_t - V_t。
# --------------------------------------------------------------------------------------
def solve_dha_counterfactual(base, p, Ahat=None, kaphat=None, dtau=None,
                             tol=1e-11, maxit=3000, damp=0.5, verbose=False):
    """在 t=0 宣布（此前未预期）的基本面变化下，求反事实路径（相对基线）。

    Ahat: (T+1,N,J) 生产率 A'_t/A_t；kaphat: (T+1,N,N,J) 贸易成本 κ'_t/κ_t；
    dtau: (T+1,M,M) 迁移成本变化 τ'_t - τ_t（对角线须为 0）。缺省为无变化。
    方程组（t = 0..T）：
      μ'_t = μ_t · exp((β D_{t+1} - Δτ_t)/ν) / Σ(·)
      D_t  = log ω̂_t + ν log Σ_k μ_t^{·k} exp((β D_{t+1}^k - Δτ_t^{·k})/ν)
      L'_{t+1} = μ'_t' L'_t，L'_0 = L_0；ω̂_t = ω'_t/ω_t 由相对基线同期均衡的精确帽子代数给出。
    终端 D_T 取稳态不动点。D_0 即各劳动力市场的终生福利变化（效用单位）。
    这与 CDP 命题 3 的“相对时间差分”写法等价：û_{t+1} = u̇'_{t+1}/u̇_{t+1} = exp(D_{t+1} - D_t)。
    """
    T1, M = base["L"].shape
    N, J = p.N, p.J
    Ahat = np.ones((T1, N, J)) if Ahat is None else _broadcast_path(Ahat, (T1, N, J))
    kaphat = None if kaphat is None else _broadcast_path(kaphat, (T1, N, N, J))
    dtau = np.zeros((T1, M, M)) if dtau is None else _broadcast_path(dtau, (T1, M, M))
    assert np.allclose(np.diagonal(dtau, axis1=1, axis2=2), 0.0), "Δτ 的对角线必须为 0"

    logmu_b = base["logmu"]
    Lp_base = employment(base["L"], base["L_for"], p)
    D = np.zeros((T1, M))
    logw = logP = None
    D_end = None
    err_eq = 1e-4
    step = _OuterIteration(damp)
    for it in range(maxit):
        D_next = np.concatenate([D[1:], D[-1:]], axis=0)
        logmu_cf = _log_softmax_rows(logmu_b + (p.beta * D_next[:, None, :] - dtau) / p.nu)
        mu_cf = np.exp(logmu_cf)
        L_cf = _forward_labor(base["L"][0], mu_cf[:-1])
        Lhat = employment(L_cf, base["L_for"], p) / Lp_base
        eq = solve_temp_eq_hat(base["pi"], base["wL"], Lhat, p, Ahat=Ahat, kaphat=kaphat,
                               logw0=logw, logP0=logP, tol=max(tol * 0.1, err_eq * 1e-3))
        logw, logP = eq["logw"], eq["logP"]
        log_omega_hat = lm_log_omega(eq["log_omega_hat"], 0.0, p)          # (T1, M)
        D_new = np.empty_like(D)
        D_new[-1] = D_end = stationary_value(log_omega_hat[-1], -p.nu * logmu_b[-1] + dtau[-1],
                                             p.beta, p.nu, V0=D_end)
        for t in range(T1 - 2, -1, -1):
            D_new[t] = log_omega_hat[t] + p.nu * _lse(
                logmu_b[t] + (p.beta * D_new[t + 1][None, :] - dtau[t]) / p.nu, axis=1)
        err = np.max(np.abs(D_new - D))
        err_eq = err
        if verbose and it % 20 == 0:
            print(f"  [DHA counterfactual] iter {it:4d}  max|ΔD| = {err:.2e}")
        if err < tol:
            break
        D = step(D, D_new, err)
    else:
        raise RuntimeError(f"DHA 反事实未收敛：max|ΔD| = {err:.2e}")
    return dict(L=L_cf, mu=mu_cf, logmu=logmu_cf, D=D_new, log_omega_hat=log_omega_hat,
                pi=eq["pi"], X=eq["X"], wL=eq["wL"], iterations=it + 1)


def welfare(cf, base, p):
    """福利变化。

    返回：
      D0:  终生价值变化 V'_0 - V_0（效用单位）；
      ce:  消费等价变化（%），即基线中每期消费乘以 (1+ce) 所带来的同等福利，log(1+ce) = (1-β) D_0；
      decomposition_gap: 用 CDP 分解式 Σ β^t [log ω̂_t - ν log(μ'^{aa}_t/μ^{aa}_t)] + β^T D_T
                         重算 D_0 后与直接解的最大差距（数值检查）。
    """
    D0 = cf["D"][0]
    T1 = cf["D"].shape[0]
    own_cf = np.diagonal(cf["logmu"], axis1=1, axis2=2)
    own_b = np.diagonal(base["logmu"], axis1=1, axis2=2)
    disc = p.beta ** np.arange(T1 - 1)
    terms = cf["log_omega_hat"][:-1] - p.nu * (own_cf[:-1] - own_b[:-1])
    D0_decomp = disc @ terms + p.beta ** (T1 - 1) * cf["D"][-1]
    return dict(D0=D0, ce=100.0 * np.expm1((1.0 - p.beta) * D0),
                decomposition_gap=float(np.max(np.abs(D0_decomp - D0))))
