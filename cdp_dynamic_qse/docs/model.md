# CDP 型动态空间一般均衡模型：设定、动态帽子代数、估计与反事实

本文档给出代码（`cdp/`）所实现模型的完整推导。模型沿用 Caliendo, Dvorkin & Parro (2019, *Econometrica*，下称 CDP) 的结构：前瞻性的劳动力在"地区 × 部门"劳动力市场之间迁移，有迁移摩擦；生产端是多部门、带投入产出联系的 Eaton–Kortum 贸易模型。所有公式都与代码一一对应，文中标注了对应函数。

---

## 1. 环境与记号

- 地区 $n, i \in \lbrace 1,\dots,N\rbrace$。前 $N_d$ 个是**国内地区**，劳动力可在其间迁移；其余是**外国**，各部门劳动力固定。
- 生产部门 $j, k \in \lbrace 1,\dots,J\rbrace$。国内劳动力市场记为 $(n,j)$，其中 $j=0$ 表示**非就业**，共 $M = N_d (J+1)$ 个。
- 时间 $t = 0, 1, 2, \dots$（季度）。期初的劳动力配置 $L_t$ 给定：先进行生产与消费（**临时均衡**），期末再做迁移决策。
- 贸易份额 $\pi^{nj,ij}_t$：地区 $n$ 在部门 $j$ 的支出中，购自地区 $i$ 的份额。代码中数组按 `pi[n, i, j]` 排列。
- **基本面** $\Theta_t = (A_t, \kappa_t, \tau_t, H, b)$：生产率、冰山贸易成本、迁移成本、结构（土地/建筑）存量、非就业者的家庭生产。
- **弹性与份额**：$\beta,\ \nu,\ \theta^j,\ \alpha^{nj},\ \gamma^{nj},\ \gamma^{nj,nk},\ \xi^n,\ \iota^n$。动态帽子代数只需要这一组参数。

## 2. 家庭：前瞻性的迁移决策

在 $(n,j)$ 的劳动者本期消费 $C^{nj}_t$，期末观察到对每个目的地 $(i,k)$ 的偏好冲击 $\epsilon^{ik}_t$（第一类极值分布，离散度参数为 $\nu$），然后选择下一期的劳动力市场：

$$
v^{nj}_t = \log C^{nj}_t + \max_{(i,k)} \left\lbrace \beta\, \mathbb{E}\big[v^{ik}_{t+1}\big] - \tau^{nj,ik} + \nu\, \epsilon^{ik}_t \right\rbrace ,
\qquad
C^{nj}_t = \begin{cases} w^{nj}_t / P^n_t, & j \ge 1 \\ b^n, & j = 0 \end{cases}
$$

记 $V^{nj}_t \equiv \mathbb{E}[v^{nj}_t]$。由极值分布的性质（常数并入 $V$，不影响任何结果）：

$$
V^{nj}_t = \log C^{nj}_t + \nu \log \sum_{(i,k)} \exp\!\left(\frac{\beta V^{ik}_{t+1} - \tau^{nj,ik}}{\nu}\right) \tag{2.1}
$$

$$
\mu^{nj,ik}_t = \frac{\exp\!\big((\beta V^{ik}_{t+1} - \tau^{nj,ik})/\nu\big)}{\sum_{(m,h)} \exp\!\big((\beta V^{mh}_{t+1} - \tau^{nj,mh})/\nu\big)} \tag{2.2}
$$

$$
L^{ik}_{t+1} = \sum_{(n,j)} \mu^{nj,ik}_t\, L^{nj}_t \tag{2.3}
$$

$1/\nu$ 就是迁移对价值差的弹性。$\tau^{nn,nn}=0$。

## 3. 生产、贸易与市场出清（临时均衡）

**技术。** 部门 $j$ 的每个中间品种类，用劳动、结构和各部门的复合中间品生产：

$$
q^{nj} = z\, A^{nj}\, \big[(h^{nj})^{\xi^n}(l^{nj})^{1-\xi^n}\big]^{\gamma^{nj}} \prod_k (M^{nj,nk})^{\gamma^{nj,nk}},
\qquad \gamma^{nj} + \sum_k \gamma^{nj,nk} = 1 .
$$

投入束的单位成本（$r$ 为结构租金）为

$$
x^{nj} = B^{nj}\big[(r^{nj})^{\xi^n}(w^{nj})^{1-\xi^n}\big]^{\gamma^{nj}} \prod_k (P^{nk})^{\gamma^{nj,nk}} . \tag{3.1}
$$

**贸易（Eaton–Kortum）。** 效率 $z$ 服从形状参数为 $\theta^j$ 的 Fréchet 分布，于是

$$
P^{nj} = \Big[\sum_i \big(x^{ij}\kappa^{nj,ij}\big)^{-\theta^j}(A^{ij})^{\theta^j\gamma^{ij}}\Big]^{-1/\theta^j},
\qquad
\pi^{nj,ij} = \frac{\big(x^{ij}\kappa^{nj,ij}\big)^{-\theta^j}(A^{ij})^{\theta^j\gamma^{ij}}}{\sum_m \big(x^{mj}\kappa^{nj,mj}\big)^{-\theta^j}(A^{mj})^{\theta^j\gamma^{mj}}} . \tag{3.2}
$$

消费者价格指数为 $P^n = \prod_j (P^{nj})^{\alpha^{nj}}$，实际工资为 $\omega^{nj} = w^{nj}/P^n$。

**市场出清。** 结构租金进入一个全球组合，地区 $n$ 分得份额 $\iota^n$，贸易不平衡由此产生：

$$
X^{nj} = \sum_k \gamma^{nk,nj} \sum_i \pi^{ik,nk} X^{ik} + \alpha^{nj} I^n,
\qquad
I^n = \sum_k w^{nk}L^{nk} + \iota^n \chi,
\qquad
\chi = \sum_{i,k} r^{ik}H^{ik} \tag{3.3}
$$

$$
w^{nj}L^{nj} = \gamma^{nj}(1-\xi^n)\sum_i \pi^{ij,nj}X^{ij},
\qquad
r^{nj}H^{nj} = \frac{\xi^n}{1-\xi^n}\, w^{nj}L^{nj} . \tag{3.4}
$$

给定 $L_t$ 与 $\Theta_t$，由 (3.1)–(3.4) 解出 $\lbrace w_t, P_t, \pi_t, X_t\rbrace$，就是**临时均衡**（`static_eq.solve_temp_eq_levels`）。

**序贯竞争均衡**：给定 $L_0$ 和 $\lbrace\Theta_t\rbrace$，找到 $\lbrace L_t, \mu_t, V_t\rbrace$ 与各期临时均衡，使 (2.1)–(2.3) 和 (3.1)–(3.4) 同时成立（`dynamics.solve_levels_path`）。**稳态**是 $L$、$\mu$、$V$ 都不再变化的序贯均衡。

## 4. 精确帽子代数（临时均衡的变化）

对任意变量记 $\hat{x} = x'/x$。结构存量固定，由 (3.4) 得 $\hat r = \hat w \hat L$，代入 (3.1)，常数 $B$ 被消去：

$$
\hat x^{nj} = (\hat L^{nj})^{\gamma^{nj}\xi^n}(\hat w^{nj})^{\gamma^{nj}}\prod_k (\hat P^{nk})^{\gamma^{nj,nk}}, \qquad
\hat P^{nj} = \Big[\sum_i \pi^{nj,ij}\big(\hat x^{ij}\hat\kappa^{nj,ij}\big)^{-\theta^j}(\hat A^{ij})^{\theta^j\gamma^{ij}}\Big]^{-1/\theta^j} \tag{4.1}
$$

$$
\pi'^{nj,ij} = \pi^{nj,ij}\Big(\frac{\hat x^{ij}\hat\kappa^{nj,ij}}{\hat P^{nj}}\Big)^{-\theta^j}(\hat A^{ij})^{\theta^j\gamma^{ij}},
\qquad
\hat w^{nj}\hat L^{nj}\,(wL)^{nj} = \gamma^{nj}(1-\xi^n)\sum_i \pi'^{ij,nj}X'^{ij} \tag{4.2}
$$

$X'$ 由 (3.3) 用新的 $\pi'$ 和收入 $I'$ 重新求解。这一系统只需要参照期的 $\pi$、$wL$、各种份额和 $\theta$，不需要 $A, \kappa, H$ 的水平值（`static_eq.solve_temp_eq_hat`）。代码测试检验了它与两次水平解之比完全一致（误差约 1e-13）。

## 5. 动态帽子代数：基线（CDP 命题 2）

定义 $\dot x_{t+1} = x_{t+1}/x_t$，以及 $\dot u^{nj}_{t+1} \equiv \exp(V^{nj}_{t+1} - V^{nj}_t)$。对相邻两期的 (2.2) 取比值，$\tau$ 被消去：

$$
\mu^{nj,ik}_{t} = \frac{\mu^{nj,ik}_{t-1}\,(\dot u^{ik}_{t+1})^{\beta/\nu}}{\sum_{(m,h)}\mu^{nj,mh}_{t-1}\,(\dot u^{mh}_{t+1})^{\beta/\nu}} \tag{5.1}
$$

对 (2.1) 做差分，并利用 (2.2) 把分母写成 $\mu_{t-1}$ 的加权和：

$$
\dot u^{nj}_{t} = \dot\omega^{nj}_{t}\Big(\sum_{(i,k)}\mu^{nj,ik}_{t-1}(\dot u^{ik}_{t+1})^{\beta/\nu}\Big)^{\nu} \tag{5.2}
$$

再加上 (2.3) 和时间差分形式的临时均衡 (4.1)–(4.2)，就构成一个闭合系统。**所需数据只有**：

| 数据 | 用途 |
|---|---|
| $L_0$：初期各劳动力市场人口 | 初始条件 |
| $\mu_{-1}$：上一期的迁移流矩阵 | 包含了迁移成本和价值水平的全部信息 |
| $\pi_0$、$wL_0$（以及 $X_0$）：贸易份额与增加值 | 临时均衡的参照点 |
| $\alpha, \gamma, \xi, \iota$ | 投入产出表、国民账户 |
| $\beta, \nu, \theta$ | 估计或取自文献（第 7 节） |

$A, \kappa, \tau, b, H$ 的水平值都不需要，这就是动态帽子代数的威力。终端条件为 $\dot u_{T+1}=1$（$T$ 足够大，经济收敛到稳态）。

**实现细节**（`dynamics.solve_dha_baseline`）：各期临时均衡都相对 $t=0$ 的观测均衡求解（$\hat L = L_t/L_0$），这与逐期链式求解在代数上等价，而且各期可以并行计算。$\dot\omega_t$ 由相邻两期的累计变化相除得到。

## 6. 反事实与福利

**信息结构。** 经济在 $t=0$ 处于基线上；$t=0$ 时（期初人口 $L_0$ 已定、迁移决策之前）宣布新的基本面路径 $\Theta'_t = \Theta_t \odot \hat\Theta_t$，此前未被预期。

**相对基线的水平差表述。** 记 $D^{nj}_t \equiv V'^{nj}_t - V^{nj}_t$，$\Delta\tau_t \equiv \tau'_t - \tau_t$。由 (2.1)–(2.2) 直接得到（`dynamics.solve_dha_counterfactual`）：

$$
\mu'^{nj,ik}_t = \frac{\mu^{nj,ik}_t \exp\!\big((\beta D^{ik}_{t+1} - \Delta\tau^{nj,ik}_t)/\nu\big)}{\sum_{(m,h)}\mu^{nj,mh}_t \exp\!\big((\beta D^{mh}_{t+1} - \Delta\tau^{nj,mh}_t)/\nu\big)} \tag{6.1}
$$

$$
D^{nj}_t = \log\hat\omega^{nj}_t + \nu \log \sum_{(i,k)} \mu^{nj,ik}_t \exp\!\big((\beta D^{ik}_{t+1} - \Delta\tau^{nj,ik}_t)/\nu\big) \tag{6.2}
$$

$$
L'_{t+1} = \mu_t'^{\top} L'_t,\quad L'_0 = L_0 \tag{6.3}
$$

其中 $\hat\omega_t = \omega'_t/\omega_t$ 由第 4 节的精确帽子代数给出，参照点是**同一期**的基线均衡 $(\pi_t, wL_t)$，$\hat L_t = L'_t/L_t$。终端的 $D_T$ 取 (6.2) 在稳态下的不动点。

- **与 CDP 命题 3 的关系。** CDP 用"相对时间差分" $\hat u_{t+1} = \dot u'_{t+1}/\dot u_{t+1}$ 表述；由定义 $\hat u_{t+1} = \exp(D_{t+1} - D_t)$，代入后在 $t \ge 1$ 与 (6.1)–(6.2) 逐式等价。用水平差 $D$ 表述的好处是：宣布冲击那一期（$t=0$）不需要特殊处理，福利 $D_0$ 可以直接读出。
- **迁移成本冲击**以 $\Delta\tau$（效用单位）输入。流量表述更直观：在价值不变时，跨地区迁移的相对概率乘以 $e^{-\Delta\tau/\nu}$。代码中的情景 3 就是这样设定的。

**福利。** $D^{nj}_0$ 是 $(n,j)$ 劳动者的终生价值变化。消费等价变化 $\delta^{nj}$ 满足"基线中每期消费都乘以 $1+\delta$ 带来同等福利"：

$$
\log(1+\delta^{nj}) = (1-\beta)\, D^{nj}_0 .
$$

对 (6.2) 用 (6.1) 在 $(i,k)=(n,j)$ 处的取值（要求 $\Delta\tau^{nj,nj}=0$），可以得到 CDP 的福利分解式：

$$
D^{nj}_0 = \sum_{t\ge 0}\beta^t\Big[\log\hat\omega^{nj}_t - \nu\log\frac{\mu'^{nj,nj}_t}{\mu^{nj,nj}_t}\Big] .
$$

第一项是实际工资的变化，第二项是**选择价值**的变化：留在原地的概率下降，说明外面的选择变好了。`dynamics.welfare` 同时用两种方法计算，二者之差作为数值检查。

## 7. 估计

### 7.1 迁移弹性（Artuç, Chaudhuri & McLaren 2010；CDP 2019）

由 (2.2)，对 $a=(n,j) \neq b=(i,k)$ 有 $\log(\mu^{ab}_t/\mu^{aa}_t) = (\beta(V^b_{t+1} - V^a_{t+1}) - \tau^{ab})/\nu$。再用 (2.1)–(2.2) 把 $V^b_{t+1}$ 写成

$$
V^b_{t+1} = \log\omega^b_{t+1} + \beta V^b_{t+2} - \nu\log\mu^{bb}_{t+1},
$$

对 $a$ 同理，并利用 $t+1$ 期的同一关系消去 $V_{t+2}$，得到只含可观测量的方程：

$$
\underbrace{\log\frac{\mu^{ab}_t}{\mu^{aa}_t}}_{y} = \underbrace{-\frac{1-\beta}{\nu}\tau^{ab}}_{\text{配对固定效应}} + \frac{\beta}{\nu}\underbrace{\big(\log\omega^{b}_{t+1} - \log\omega^{a}_{t+1}\big)}_{x_1} + \beta\,\underbrace{\log\frac{\mu^{ab}_{t+1}}{\mu^{bb}_{t+1}}}_{x_2} + e^{ab}_t \tag{7.1}
$$

- 完全预见下 $e=0$。现实数据里，$e$ 包含预期误差和测量误差。
- **做法**（`estimation.estimate_migration_elasticity`）：固定 $\beta$（CDP 取 0.99），把 $y - \beta x_2$ 对 $x_1$ 回归，吸收配对固定效应。工资的测量误差会使 OLS 衰减，$\nu$ 因而被高估，所以用 $t-1$ 期的工资差作工具变量做 2SLS。标准误按配对聚类，$\nu$ 的标准误用 delta 方法计算。
- **小流量问题**：抽样得到的小迁移流取对数后有 Jensen 偏误，零流量又只能剔除（这等于按结果选样）。因此只保留样本期平均迁移份额不低于阈值（默认 $3\times 10^{-4}$）的配对。这一筛选只依赖配对层面的信息，由配对固定效应吸收。
- 非就业者没有工资数据，这不要紧：$b^n$ 不随时间变化，被配对固定效应吸收。

### 7.2 贸易弹性（PPML 面板引力方程）

由 (3.2)，双边贸易额 $X^{nj,ij}_t = \pi^{nj,ij}_t X^{nj}_t$ 满足

$$
X^{nj,ij}_t = \exp\big(\mathrm{FE}^{nj}_t + \mathrm{FE}^{ij}_t + \mathrm{FE}^{ni,j}\big)\,(1 + f^{nj,ij}_t)^{-\theta^j}, \tag{7.2}
$$

其中模拟设定 $\log\kappa^{nj,ij}_t = \log(1+f^{nj,ij}_t) + \delta^j\log(1+d_{ni}) + \text{边境} + \eta^{nj,ij}$，$f$ 是可观测的从价运费。进口地×年份和出口地×年份的固定效应吸收多边阻力；配对固定效应吸收距离、边境和不可观测的 $\eta$。$\theta^j$ 由配对内部运费的**时间变化**识别，用 PPML 估计，标准误按配对聚类（`estimation.estimate_trade_elasticity`）。

> 一个教训：最初版本只用截面，并以距离、边境作为控制变量，结果 $\theta$ 严重有偏。原因是截面上只有约 90 个地区对，不可观测的贸易成本 $\eta$ 成为主导误差；而且截面在各年重复出现，未聚类的标准误还低估了不确定性。改用面板加配对固定效应后，偏误消失。真实数据中常见的替代做法是用关税变化（Caliendo & Parro 2015），或直接取文献值。

### 7.3 份额参数：直接从数据读取

- $\alpha^{nj}$：最终消费支出结构；$\gamma^{nj}, \gamma^{nj,nk}$：投入产出表；$\xi^n$：增加值中结构的份额。
- $\iota^n$ 由贸易不平衡反推（`static_eq.calibrate_iota`）：最终支出 $F^{nj} = X^{nj} - \sum_k\gamma^{nk,nj}Y^{nk}$，地区收入 $I^n = \sum_j F^{nj}$，于是 $\iota^n = (I^n - \sum_j w^{nj}L^{nj})/\chi$。

## 8. 数值算法

| 层次 | 不动点 | 做法 |
|---|---|---|
| 价格 | (4.1) 中的 $P$ | 直接迭代（投入产出矩阵的谱半径 < 1，是压缩映射）；精度随外层误差自适应收紧 |
| 临时均衡 | 工资 $w$ | 按 $1/(1+\theta^j\gamma^{nj})$ 缩放步长，相当于对角近似的牛顿步（全步长会因 $\theta$ 大而超调发散），再加 Anderson 加速 |
| 动态 | 价值路径 $V$ / $\log\dot u$ / $D$ | 与 CDP 的算法相同：猜价值路径 → 迁移份额 → 人口 → 临时均衡 → 新价值路径。阻尼加 Anderson 加速。在人口路径上迭代会出现振荡，在价值路径上迭代则稳定 |
| 稳态 | (2.1) 与 (6.2) 的终端 | 牛顿法，雅可比矩阵为 $I-\beta\mu$ |

各期的临时均衡互相独立，代码把它们沿时间维度批量并行求解。完整流程（T=400 季度、32 个劳动力市场、10 个地区 × 3 个部门）的每次动态求解约 10–20 秒。

## 9. 模拟设计（数据生成过程）

- 8 个国内地区：R1–R3 为沿海，R4–R8 为内陆。另有 2 个外国：F1 是外国 A，F2 是世界其他地区。3 个部门：农业、制造业、服务业。季度频率，$\beta = 0.99$，$\nu = 5.34$（CDP 的季度值），$\theta = (8, 5, 4)$。
- 贸易成本 = 运费 × 距离 × 边境 × 不可观测成分。服务业的距离弹性最大，最难贸易。
- 迁移成本 = 跨地区的固定成本 + 距离项 + 换部门成本（进出非就业的成本较低），再加随机扰动。
- **历史期**（前 40 个季度）：生产率服从 AR(1)，运费也随时间波动；此后基本面冻结。$t=0$ 的人口分布偏离稳态，所以观测期仍处于转移动态之中。
- **研究者观测到的数据**：
  - $t_{\text{obs}}$ 期的截面 $(L, \mu_{-1}, \pi, X, wL)$；
  - 历史迁移面板：每季度按人口比例抽取 400 万人统计去向，属于多项分布抽样误差；
  - 历史实际工资：加入 $N(0, 0.03^2)$ 的测量误差；
  - 10 年的年度双边贸易额：加入 10% 的对数正态测量误差，并附有运费数据。
