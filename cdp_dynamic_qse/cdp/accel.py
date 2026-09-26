"""Anderson 加速（type-II），用于加速各层不动点迭代 x = g(x)。支持前导批量维度。"""
import numpy as np


class Anderson:
    """对批量不动点问题做 Anderson 混合。

    用法：每次迭代算出 gx = g(x) 后调用 x = acc.update(x, gx)。
    x, gx 形状为 (B, n)；每个批量元素各自保存最近 m 步的历史并各自求最小二乘系数。
    """

    def __init__(self, m=6, mix=1.0, reg=1e-12):
        self.m, self.mix, self.reg = m, mix, reg
        self.x_prev = self.f_prev = None
        self.dX, self.dF = [], []

    def update(self, x, gx):
        f = gx - x
        if self.x_prev is not None:
            self.dX.append(x - self.x_prev)
            self.dF.append(f - self.f_prev)
            if len(self.dX) > self.m:
                self.dX.pop(0)
                self.dF.pop(0)
        self.x_prev, self.f_prev = x.copy(), f.copy()
        if not self.dX:
            return x + self.mix * f
        dX = np.stack(self.dX, axis=-1)            # (B, n, k)
        dF = np.stack(self.dF, axis=-1)
        G = np.einsum("bnk,bnl->bkl", dF, dF)
        scale = np.trace(G, axis1=1, axis2=2)[:, None, None] / G.shape[-1] + 1e-300
        G = G + self.reg * scale * np.eye(G.shape[-1])[None]
        gam = np.linalg.solve(G, np.einsum("bnk,bn->bk", dF, f)[..., None])[..., 0]
        return x + self.mix * f - np.einsum("bnk,bk->bn", dX + self.mix * dF, gam)

    def reset(self):
        self.x_prev = self.f_prev = None
        self.dX, self.dF = [], []
