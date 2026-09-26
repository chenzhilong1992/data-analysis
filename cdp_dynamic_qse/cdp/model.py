"""模型参数、基本面与劳动力市场索引。

记号约定（与 docs/model.md 一致）：
- 地区 n, i = 0..N-1，其中前 n_dom 个为国内地区（劳动力可迁移），其余为外国（劳动力固定）。
- 部门 j, k = 0..J-1（生产部门）。
- 国内劳动力市场 a = (n, s)，s = 0 表示非就业，s = 1..J 表示在部门 s-1 就业；
  扁平索引 a = n * (J + 1) + s，共 M = n_dom * (J + 1) 个。
- 贸易数组一律按 [进口地 n, 出口地 i, 部门 j] 排列，即 pi[n, i, j] = π^{nj,ij}。
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    """结构参数：动态帽子代数（DHA）只需要这些弹性与份额，不需要基本面水平值。"""

    beta: float                # 季度折现因子 β
    nu: float                  # 迁移冲击离散度 ν（迁移弹性 = 1/ν）
    theta: np.ndarray          # (J,) 贸易弹性 θ^j
    alpha: np.ndarray          # (N, J) 最终消费支出份额 α^{nj}
    gamma_va: np.ndarray       # (N, J) 增加值占总产出份额 γ^{nj}
    gamma_io: np.ndarray       # (N, J, J) γ^{nj,nk}：地区 n 部门 j 对部门 k 中间品的支出份额
    xi: np.ndarray             # (N,) 增加值中结构（土地/建筑）份额 ξ^n
    iota: np.ndarray           # (N,) 地区 n 在全球结构租金组合中的份额 ι^n
    n_dom: int                 # 国内地区数
    region_names: list = field(default_factory=list)
    sector_names: list = field(default_factory=list)

    @property
    def N(self):
        return self.alpha.shape[0]

    @property
    def J(self):
        return self.alpha.shape[1]

    @property
    def n_for(self):
        return self.N - self.n_dom

    @property
    def M(self):
        return self.n_dom * (self.J + 1)

    def check(self, tol=1e-10):
        """检查份额的加总约束。"""
        assert np.allclose(self.alpha.sum(axis=1), 1.0, atol=tol), "α 每行之和应为 1"
        assert np.allclose(self.gamma_va + self.gamma_io.sum(axis=2), 1.0, atol=tol), \
            "γ^{nj} + Σ_k γ^{nj,nk} 应为 1"
        assert np.isclose(self.iota.sum(), 1.0, atol=tol), "ι 之和应为 1"
        assert np.all((self.xi > 0) & (self.xi < 1))
        assert self.theta.shape == (self.J,)


@dataclass
class Fundamentals:
    """基本面水平值：只在“真实”数据生成过程（水平解）中使用；DHA 不需要它们。"""

    A: np.ndarray       # (N, J) 或 (T, N, J) 生产率
    kappa: np.ndarray   # (N, N, J) 或 (T, N, N, J) 冰山贸易成本 κ^{nj,ij}，κ^{nj,nj} = 1
    H: np.ndarray       # (N, J) 结构（土地/建筑）存量，固定
    tau: np.ndarray     # (M, M) 或 (T, M, M) 迁移成本（效用单位），对角线为 0
    b: np.ndarray       # (n_dom,) 非就业者的家庭生产（实际消费）
    L_for: np.ndarray   # (n_for, J) 外国各部门劳动力，固定


def lm_reshape(x_dom, p):
    """(..., M) -> (..., n_dom, J+1)。"""
    return x_dom.reshape(x_dom.shape[:-1] + (p.n_dom, p.J + 1))


def employment(L_dom, L_for, p):
    """把国内劳动力市场人口 (..., M) 与外国劳动力拼成生产用就业 (..., N, J)。"""
    L_dom_emp = lm_reshape(L_dom, p)[..., 1:]
    L_for_b = np.broadcast_to(L_for, L_dom_emp.shape[:-2] + L_for.shape)
    return np.concatenate([L_dom_emp, L_for_b], axis=-2)


def lm_log_omega(log_omega_emp, log_omega_ne, p):
    """把生产部门实际工资 (..., N, J) 与非就业实际消费 (..., n_dom) 拼成劳动力市场向量 (..., M)。"""
    emp = log_omega_emp[..., :p.n_dom, :]
    ne = np.broadcast_to(log_omega_ne, emp.shape[:-1])[..., None]
    return np.concatenate([ne, emp], axis=-1).reshape(emp.shape[:-2] + (p.M,))


def lm_labels(p):
    """劳动力市场标签，如 'R1-非就业'、'R1-制造业'。"""
    names = ["非就业"] + list(p.sector_names)
    return [f"{p.region_names[n]}-{names[s]}" for n in range(p.n_dom) for s in range(p.J + 1)]
