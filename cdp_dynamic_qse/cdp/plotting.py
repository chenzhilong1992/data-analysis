"""作图（matplotlib，静态 PNG，浅色主题）。配色取自固定顺序的分类色板，文字一律用中性墨色。"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 中文字体无粗体时的提示

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # 蓝、橙、青、黄（固定顺序）


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
                            "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "figure.dpi": 150,
    })


def _zero_line(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0, color=AXIS, linewidth=0.8, zorder=1)


def _grouped_bars(ax, labels, series, names, width=0.36):
    """分组柱状图：series 为若干等长数组。柱之间留出与背景同色的细缝。"""
    x = np.arange(len(labels))
    k = len(series)
    for s, (vals, name) in enumerate(zip(series, names)):
        ax.bar(x + (s - (k - 1) / 2) * width, vals, width=width, color=SERIES[s], label=name,
               edgecolor=SURFACE, linewidth=1.2, zorder=2)
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    _zero_line(ax)


def fig_validation(lev_L, dha_L, p, regions, fname, max_err):
    """基线路径：已知基本面的水平解（线）与只用观测数据的 DHA（点）。"""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    t = np.arange(lev_L.shape[0])
    pop_lev = lev_L.reshape(len(t), p.n_dom, -1).sum(axis=2)
    pop_dha = dha_L.reshape(len(t), p.n_dom, -1).sum(axis=2)
    marks = t[::16]
    for s, n in enumerate(regions):
        ax.plot(t / 4, 100 * pop_lev[:, n], color=SERIES[s], label=f"{p.region_names[n]}：水平解", zorder=2)
        ax.plot(marks / 4, 100 * pop_dha[marks, n], "o", color=SERIES[s], markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"{p.region_names[n]}：DHA", zorder=3)
    ax.set_xlabel("年（t=0 为观测年份）")
    ax.set_ylabel("地区人口占国内总人口（%）")
    ax.set_title("基线转移路径：DHA 只用观测数据即可复现水平解", loc="left")
    ax.set_title(f"全部 {p.M} 个劳动力市场、{len(t)} 期的最大绝对误差：{max_err:.1e}",
                 loc="right", fontsize=8, fontweight="normal", color=INK2)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def fig_estimation(mc, truth, sector_names, fname):
    """蒙特卡洛：迁移弹性（OLS vs IV）与三个部门的贸易弹性。"""
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
    ax = axes[0]
    bins = np.linspace(min(mc["nu_iv"].min(), mc["nu_ols"].min()) - 0.1,
                       max(mc["nu_iv"].max(), mc["nu_ols"].max()) + 0.1, 28)
    ax.hist(mc["nu_ols"], bins=bins, color=SERIES[1], alpha=0.85, label="OLS", edgecolor=SURFACE, linewidth=0.8)
    ax.hist(mc["nu_iv"], bins=bins, color=SERIES[0], alpha=0.85, label="2SLS（滞后工资作工具）",
            edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(truth["nu"], color=INK, linewidth=1.0)
    ax.set_title(f"迁移冲击离散度 ν（真值 {truth['nu']:.2f}）", loc="left")
    ax.set_ylabel("重复次数")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    for j, name in enumerate(sector_names):
        ax = axes[j + 1]
        ax.hist(mc[f"theta_{j}"], bins=20, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(truth["theta"][j], color=INK, linewidth=1.0)
        ax.set_title(f"贸易弹性 θ：{name}（真值 {truth['theta'][j]:g}）", loc="left")
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.text(0.01, -0.08, "竖线为真值；ν 的 OLS 受工资测量误差影响而高估，2SLS 用 t-1 期工资差作工具变量。",
             fontsize=8, color=INK2, ha="left")
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def fig_scenario(title, t_years, paths, path_names, path_ylabel, lr_labels, lr_vals, lr_ylabel,
                 welfare_labels, welfare_series, welfare_names, fname):
    """单个反事实：(A) 聚合结果的转移路径；(B) 各地区长期变化；(C) 各地区福利（真实参数 vs 估计参数）。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    ax = axes[0]
    for s, (y, name) in enumerate(zip(paths, path_names)):
        ax.plot(t_years, y, color=SERIES[s], label=name)
    _zero_line(ax)
    ax.set_xlabel("冲击宣布后的年数")
    ax.set_ylabel(path_ylabel)
    ax.set_title("A. 转移路径（相对基线）", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    ax = axes[1]
    ax.bar(np.arange(len(lr_labels)), lr_vals, width=0.6, color=SERIES[0], edgecolor=SURFACE, zorder=2)
    ax.set_xticks(np.arange(len(lr_labels)), lr_labels)
    ax.grid(axis="x", visible=False)
    _zero_line(ax)
    ax.set_ylabel(lr_ylabel)
    ax.set_title("B. 长期（新稳态）变化", loc="left")
    ax = axes[2]
    _grouped_bars(ax, welfare_labels, welfare_series, welfare_names)
    ax.set_ylabel("消费等价福利变化（%）")
    ax.set_title("C. 福利：真实参数 vs 估计参数", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
