# Hasenbusch stout-smear HMC 调试报告（2026-06-05）

## 结论状态

当前目标尚未完成。已经把 12x32 `t=1` 从容易 NaN/崩溃推进到两个目标组态 `cfg1000/cfg10001` 均可完成 `stop=20`；其中 `cfg10001` 的积分器加密对照 `outer=14/inner=4` 也已完成 `20/20`，接受率和 `Delta_E` 尾部明显改善。但还没有证明 `stop=1000` 能稳定运行，也没有证明 `Delta_E` 已收敛到可接受范围。

本轮最重要的代码修复是：在 symmetric odd-odd Hasenbusch force 分支中，shifted ratio 的 `D_prec` 临时设置 `invert_param.mu = mu_prec` 后，底层 force helper 必须读取当前活动的 `invert_param.mu`，不能继续使用构造对象时的 `self.mu`。修复后，ratio 分支有限差分闭合明显改善。

## 修改文件

- `pyquda_core/pyquda/action/clover_hasenbusch.py`
  - 修复 `_add_sym_oddodd_chroma_deriv_force` 与 `_add_sym_oddodd_quda_norm_deriv_force` 中 shifted force 使用的 `mu`。
  - 关键点：用 `active_mu = self.invert_param.mu` 作为当前分支的 shift，而不是固定的 `self.mu`。
  - 备份文件：`pyquda_core/pyquda/action/clover_hasenbusch.py.codex_bak_active_mu_20260605`
- `tests/hasenbusch_operation/scripts/ratio_branch_force_fd_12x32.py`
  - 新增诊断脚本，用固定 pseudofermion、固定随机 momentum 方向，对单个 Hasenbusch ratio interval 的四个分支分别做 action 有限差分，并与 force 投影比较。
  - 输出：`csv/ratio_branch_force_fd_*.csv`

已有但本轮继续使用的实验脚本：

- `tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32.py`
  - 12x32 Hasenbusch stout-smear HMC driver。
  - 支持 `cfg=1000/10001`、`t_boundary`、trajectory length、inner/outer 积分器步数、monomial 布局、ratio/cancel force mode 等参数。
- `tests/hasenbusch_operation/scripts/force_check_hasenbusch_12x32.py`
  - 固定 pseudofermion 后做整体 monomial force 有限差分检查。

## 语法检查

```bash
python3 -m py_compile pyquda_core/pyquda/action/clover_hasenbusch.py
```

结果：通过。

## force 有限差分校验

### cfg1000，active_mu 修复后

`ratio_only`，`pc_quda_norm`，`matpc_type=odd_odd`，`cancel_constdet=yes`，`eps=3e-5`：

| seed | 有限差分 | force 投影 | 同号相对差 |
|---:|---:|---:|---:|
| 10086 | 1.2519924591 | 1.2819429575 | 0.02336 |
| 10087 | -0.5260575563 | -0.4984666968 | 0.02759 |
| 10088 | 0.7396990744 | 0.7428226908 | 0.00312 |

对比修复前，seed10086 曾约为 `FD=1.2658, force=2.9776`，说明 active `mu` bug 是实质问题。

`full` 与 `no_ratio` 的 force convention 当前表现为反号闭合更好：

| subset | seed | 有限差分 | force 投影 | `FD + force` | 反号相对差 |
|---|---:|---:|---:|---:|---:|
| no_ratio | 10086 | -408.8226746 | 409.6591798 | 0.8365052 | 约 0.0020 |
| full | 10086 | -613.3403745 | 615.4932065 | 2.1528320 | 约 0.0035 |

注意：`ratio_only` 和 `full/no_ratio` 的符号口径不同，报告中不能简单只看 `FD-force`。

### cfg10001，active_mu 修复后

`ratio_only` 三个 seed 都是同号闭合：

| seed | 有限差分 | force 投影 | `FD-force` | 同号相对差 |
|---:|---:|---:|---:|---:|
| 10086 | -0.1098184536 | -0.0968816992 | -0.0129367543 | 0.01294 |
| 10087 | -0.1002762777 | -0.0902995381 | -0.0099767396 | 0.00998 |
| 10088 | -0.3558738778 | -0.3618572646 | 0.0059833868 | 0.00598 |

`full` seed10086 仍表现为反号闭合较好：

| subset | seed | 有限差分 | force 投影 | `FD + force` | 反号相对差 |
|---|---:|---:|---:|---:|---:|
| full | 10086 | 142.8296556696 | -141.6391741801 | 1.1904814895 | 0.00833 |

解释：`full` 剩余约 0.8% 的反号相对差，可能仍会在长 HMC 中积累为可见 `Delta_E`。

## ratio 分支级诊断

运行：

```bash
python3 scripts/ratio_branch_force_fd_12x32.py   --cfg 1000 --seed 10086   --run-id ratio_branch_all_active_mu_seed10086_20260605   --ratio-index -1 --eps 3e-5
```

每个 Hasenbusch interval 有四个分支：

- `Dprec_phi_X_PLUS`
- `D_X_Y_MINUS`
- `D_Y_X_PLUS`
- `Dprec_X_phi_MINUS`

结果汇总：

| ratio interval | 最大单分支相对差 | 四分支 signed 有限差分和 | 四分支 force 和 | 总相对差 |
|---|---:|---:|---:|---:|
| has0_0_to_0.0005 | 0.003374 | 0.0004394678 | 0.0004406495 | 1.18e-6 |
| has1_0.0005_to_0.0015 | 0.001169 | -0.0123021891 | -0.0122745637 | 2.76e-5 |
| has2_0.0015_to_0.0035 | 0.001809 | -0.0587742155 | -0.0587754997 | 1.28e-6 |
| has3_0.0035_to_0.0075 | 0.001708 | -0.1977498565 | -0.1976219525 | 1.28e-4 |
| has4_0.0075_to_0.0155 | 0.001551 | 1.5493006989 | 1.5501743054 | 5.64e-4 |

判断：active `mu` 修复后，ratio 四分支结构本身已经基本闭合。剩余 HMC `Delta_E` 偏大不应继续主要怀疑 has4 分支公式，而应优先看 full monomial 中的其他项、stout force 回传累计误差、logdet/cancel/strange 符号口径，以及积分器布局。

## HMC 短程验证

所有 HMC 均为：

- `t_boundary=-1`
- `trajectory_length=1.0`
- `hasenbusch_mu=0,0.0005,0.0015,0.0035,0.0075,0.0155`
- `ratio_force_mode=pc_quda_norm`
- `cancel_force_mode=pc_quda_norm`
- `matpc_type=odd_odd`
- `cancel_constdet=yes`
- `placement=cancel_inner`
- `integrator_outer=O4Nf5Ng0V`
- `integrator_inner=O4Nf5Ng0V`

### 原步数：outer=7, inner=2

| run | cfg | stop | 接受率 | plaquette 第一分量初值 | 末值 | 波动范围 | Delta_E 平均值 | Delta_E 最大绝对值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke3 | 1000 | 3 | 3/3 | 0.5355789712 | 0.5354211459 | 0.0003577275 | -1.125715 | 1.759820 |
| stop10 | 1000 | 10 | 6/10 | 0.5355789712 | 0.5340273701 | 0.0017514919 | 0.103993 | 2.337793 |
| smoke3 | 10001 | 3 | 1/3 | 0.5354183635 | 0.5361636621 | 0.0007452985 | 3.601042 | 4.871594 |

### 加倍步数：outer=14, inner=4

| run | cfg | stop | 接受率 | plaquette 第一分量初值 | 末值 | 波动范围 | Delta_E 平均值 | Delta_E 最大绝对值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke3 | 10001 | 3 | 1/3 | 0.5354183635 | 0.5361635559 | 0.0007451923 | 3.217362 | 4.395303 |
| stop5 | 1000 | 5 | 4/5 | 0.5355789712 | 0.5355146636 | 0.0003588135 | -0.734429 | 2.178096 |

判断：加倍 O4 步数没有从根本上降低 `Delta_E`，但单条 trajectory 成本显著升高。因此不建议直接靠继续加步数推进 `stop=1000`；应继续定位 full force 剩余偏差和积分器/monomial 布局差异。


## full force 逐 monomial 审计

运行：

```bash
python3 scripts/force_check_hasenbusch_12x32.py \
  --monomial-subset full \
  --ratio-force-mode pc_quda_norm \
  --cancel-force-mode pc_quda_norm \
  --matpc-type odd_odd \
  --cancel-constdet yes \
  --eps 3e-5 \
  --tol-override 1e-10 --maxiter-override 5000 \
  --diagnose-monomial-fd
```

说明：`monomial_0` 是 `CloverWilsonLogDetAction`，`monomial_7` 是 strange `CloverWilsonAction`。ratio monomial 的单项符号 convention 与 full/no_ratio 的总符号 convention 不完全一样，因此逐项表主要用于定位贡献来源，不能孤立地把某个 ratio 项的 `FD+force` 当成 bug。

### cfg1000

| monomial | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---|---:|---:|---:|---:|
| logdet (`monomial_0`) | -8.7746182544 | 8.8401846704 | 0.0655664160 | 0.00742 |
| has_cancel | -226.830205065 | 227.975589315 | 1.1453842494 | 0.00502 |
| has0 | -0.0005277495 | -0.0015857464 | -0.0021134959 | 0.00211 |
| has1 | -0.0149778013 | -0.0127063540 | -0.0276841553 | 0.02768 |
| has2 | -0.0444949061 | -0.0444512036 | -0.0889461098 | 0.08895 |
| has3 | 0.4052223327 | 0.4100363626 | 0.8152586954 | 0.81526 |
| has4 | 0.1845066436 | 0.1838302805 | 0.3683369241 | 0.36834 |
| strange (`monomial_7`) | -378.251123766 | 378.142309095 | -0.1088146706 | 0.00029 |
| total | -613.326218566 | 615.493206419 | 2.1669878533 | 0.00352 |

### cfg10001

| monomial | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---|---:|---:|---:|---:|
| logdet (`monomial_0`) | 69.5913027812 | -69.6922050455 | -0.1009022643 | 0.00145 |
| has_cancel | 125.653997141 | -126.283842604 | -0.6298454627 | 0.00499 |
| has0 | 0.0280473614 | -0.0004580085 | 0.0275893529 | 0.02759 |
| has1 | 0.0046449713 | 0.0065112298 | 0.0111562012 | 0.01116 |
| has2 | 0.0191144257 | 0.0162512886 | 0.0353657143 | 0.03537 |
| has3 | -0.0641846176 | -0.0540702477 | -0.1182548653 | 0.11825 |
| has4 | 0.6484866996 | 0.6598150716 | 1.3083017712 | 1.30830 |
| strange (`monomial_7`) | -53.0563560763 | 53.7088243836 | 0.6524683072 | 0.01215 |
| total | 142.825052687 | -141.639173932 | 1.1858787544 | 0.00830 |

判断：full 残差不是单一 monomial 的明显爆炸，而是 cancel、strange、ratio interval 的抵消后还剩系统性差异。cfg10001 的 strange/cancel/has4 抵消尤其明显。

## full force eps 扫描

目的：排除 `eps=3e-5` 有限差分步长导致的假残差。

| cfg | eps | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 1e-4 | -613.327978645 | 615.493206420 | 2.1652277756 | 0.00352 |
| 1000 | 3e-5 | -613.326230086 | 615.493206420 | 2.1669763338 | 0.00352 |
| 1000 | 1e-5 | -613.307754975 | 615.493206420 | 2.1854514453 | 0.00355 |
| 10001 | 1e-4 | 142.807429656 | -141.639173806 | 1.1682558504 | 0.00818 |
| 10001 | 3e-5 | 142.834649887 | -141.639173806 | 1.1954760809 | 0.00837 |
| 10001 | 1e-5 | 142.863707151 | -141.639173806 | 1.2245333452 | 0.00857 |

判断：残差随 eps 基本稳定，不是有限差分步长噪声。

## monomial placement 验证

在 cfg10001 上测试两个替代布局，其他参数不变：

- `all_fermion_outer`：inner 只放 gauge，所有 fermion 放 outer。
- `all_outer`：gauge、logdet、cancel、ratio、strange 全放 outer。

结果：

| placement | stop | 状态 | 接受率 / Delta_E |
|---|---:|---|---|
| all_fermion_outer | 3 | 失败 | trajectory 0 后 solver residual NaN，MPI_ABORT |
| all_outer | 3 | 完成但不可用 | 0/3 接受，`Delta_E` 约 `1.04e10`, `2.49e8`, `8.94e9` |

判断：替代 placement 没有改善。当前最可用布局仍是 `cancel_inner`，但它也只能短程运行，尚未达到 `t=1, stop=1000` 目标。


## subset force 独立闭合检查

为了判断 full force 残差来源，分别检查 `logdet_only`、`cancel_only`、`strange_only`。所有检查使用 `eps=3e-5`、`matpc_type=odd_odd`、`cancel_constdet=yes`。

| cfg | subset / mode | 有限差分 | force 投影 | `FD+force` | 反号相对差 | 判断 |
|---:|---|---:|---:|---:|---:|---|
| 1000 | logdet_only | -8.774618254 | 8.840184670 | 0.065566416 | 0.00742 | 可用但有约 0.7% 残差 |
| 1000 | cancel_only pc_quda_norm | -226.831487574 | 227.975589316 | 1.144101742 | 0.00502 | 三种 cancel mode 中最好 |
| 1000 | cancel_only pc_constdet | -226.830204095 | 241.271834087 | 14.441629992 | 0.05986 | 明显更差 |
| 1000 | cancel_only pc_pair | -226.829642391 | -8134.26705955 | -8361.09670194 | 1.02789 | 不可用 |
| 1000 | strange_only | -267.740363779 | 267.393427518 | -0.346936261 | 0.00130 | 较好 |
| 10001 | logdet_only | 69.591302781 | -69.692205046 | -0.100902264 | 0.00145 | 较好 |
| 10001 | cancel_only pc_quda_norm | 125.654008783 | -126.283842604 | -0.629833821 | 0.00499 | 三种 cancel mode 中最好 |
| 10001 | cancel_only pc_constdet | 125.654007813 | 205.671890407 | 331.325898220 | 1.61094 | 不可用 |
| 10001 | cancel_only pc_pair | 125.653939904 | -8137.70966436 | -8012.05572446 | 0.98456 | 不可用 |
| 10001 | strange_only | -280.336012565 | 279.951726127 | -0.384286438 | 0.00137 | 较好 |

结论：`cancel_force_mode=pc_quda_norm` 是当前可用的 cancel force mode；`pc_constdet` 虽然名字更接近 const-det，但有限差分明显不闭合，不能直接替换当前 HMC。full 残差来自多个小残差的累计和抵消，而不是某个 mode 简单选错。

## cancel_sample_scale=sqrt(0.5) 验证

Chroma multihasen cancel 常见 `sqrt(0.5)` refresh scale。当前默认一直用 `1.0`，因此补测：

```bash
--cancel-sample-scale 0.7071067811865476
```

### HMC 短跑

| cfg | stop | 接受率 | Delta_E 值 | plaquette 第一分量首末变化 | 判断 |
|---:|---:|---:|---|---:|---|
| 10001 | 3 | 1/3 | 3.59395, 0.59294, 1.92505 | +0.00158538 | 比 scale=1 的 `4.71766,1.21388,4.87159` 改善，但仍不够 |
| 1000 | 5 | 4/5 | -1.04720, -1.90554, -0.03951, -0.67784, 0.55020 | +0.00145084 | 可跑，但 plaquette 明显上移，仍需更长验证 |

### full force check

| cfg | cancel_sample_scale | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 1.0 | -613.340374548 | 615.493206463 | 2.152831915 | 0.00350 |
| 1000 | sqrt(0.5) | -499.917624984 | 501.505411419 | 1.587786435 | 0.00317 |
| 10001 | 1.0 | 142.829655670 | -141.639174180 | 1.190481489 | 0.00833 |
| 10001 | sqrt(0.5) | 80.004796231 | -78.497252435 | 1.507543795 | 0.01884 |

判断：`sqrt(0.5)` 对 HMC `Delta_E` 有改善迹象，尤其 cfg10001，但它没有改善 cfg10001 的 full force 闭合。因此不能简单把默认值改成 `sqrt(0.5)` 并宣称修复；它只是后续 HMC 参数矩阵中的候选。

## 二阶积分器与 stop10 复核

为了验证“现有积分器组合是否合理”，在 cfg10001 上额外比较了二阶积分器组合，并把 `cancel_sample_scale=sqrt(0.5)` 的短跑从 `stop=3` 延长到 `stop=10`。其他核心参数保持不变：`trajectory_length=1.0`、`placement=cancel_inner`、`ratio_force_mode=pc_quda_norm`、`cancel_force_mode=pc_quda_norm`、`matpc_type=odd_odd`、`cancel_constdet=yes`。

| run | 积分器 | stop | 接受率 | Delta_E 值 | Delta_E 最大绝对值 | plaquette 第一分量首末变化 | 判断 |
|---|---|---:|---:|---|---:|---:|---|
| smoke3_o2nf1_cancelhalf | `O2Nf1Ng0V` | 3 | 0/3 | `2.30e7`, `1.65e8`, `3.53e7` | `1.65e8` | 0 | 完全不可用，全部拒绝 |
| smoke3_o2nf2_cancelhalf | `O2Nf2Ng0V` | 3 | 1/3 | `59.98`, `0.408`, `1.254` | `59.98` | +0.001538 | 明显差于 O4，第一步能量误差过大 |
| stop10_cancelhalf | `O4Nf5Ng0V` | 10 | 4/10 | `3.575`, `0.594`, `1.946`, `-0.598`, `2.380`, `1.182`, `-3.442`, `2.381`, `17.229`, `-2.357` | `17.229` | +0.002421 | 能跑完，但 `Delta_E` 仍不收敛 |

判断：当前已经试过的二阶积分器组合不能替代 `O4Nf5Ng0V`；`O2Nf1Ng0V` 会把 `Delta_E` 推到 `1e7-1e8` 量级，`O2Nf2Ng0V` 第一条 trajectory 也达到约 `60`。`O4Nf5Ng0V + cancel_sample_scale=sqrt(0.5)` 是目前短跑里相对不坏的组合，但 cfg10001 的 `stop=10` 已出现 `Delta_E=17.229`，且 plaquette 第一分量首末变化为 `+0.002421`，不能证明长期稳定。当前不应启动 `t=1, stop=1000` 长跑。

## 当前风险排序

1. `full` force 仍有约 0.3%-0.8% 的闭合偏差，可能足以在 `t=1` 长 HMC 中造成较大 `Delta_E`。
2. `ratio_only` 已基本闭合，但 `full/no_ratio` 与 `ratio_only` 的符号 convention 不一致，说明 force 输出到 HMC momentum 的符号路径还需要更系统地统一验证。
3. `logdet + cancel + strange + stout force reverse` 的组合贡献还没有逐项完成 finite-difference 闭合审计。
4. 当前 `O4Nf5Ng0V(outer=7, inner=2)` 对 cfg10001 接受率过低；`outer=14, inner=4` 只轻微改善，成本过高。

## 下一步建议

1. 对 `full` monomial 做逐项 finite-difference 审计，尤其区分：logdet、cancel、每个 ratio、strange、stout reverse 后的 thin-link force。
2. 把 `force_dot_direction` 的符号 convention 从单 monomial 到 HMC momentum 更新路径统一写成测试，避免 `ratio_only` 和 `full/no_ratio` 分别用不同解释。
3. 继续定位 `logdet_only` 与 `strange_only` 的剩余有限差分残差；这两个项单独看已是小残差，但在 full force 中会与 cancel/ratio 项抵消后放大相对影响。
4. 继续检查 `HMC._stoutSmearReverse()` 中 smeared-link force 到 thin-link momentum 的累计口径，确认每个 monomial 的 force 都在同一个 resident smeared gauge 上计算并完整回传。
5. 只有在 cfg1000/cfg10001 的 `stop=20` 或更短门槛上表现出稳定接受率和较小 `Delta_E` 后，才启动 `stop=1000`。

## 产物路径

- 报告：`tests/hasenbusch_operation/report/hasenbusch_force_debug_20260605.md`
- HMC 日志：`tests/hasenbusch_operation/logs/hmc_hasenbusch_*active_mu*.log`
- force check CSV：`tests/hasenbusch_operation/csv/force_check_*active_mu*.csv`
- ratio branch CSV：`tests/hasenbusch_operation/csv/ratio_branch_force_fd_*active_mu*.csv`
- HMC stdout：`tests/hasenbusch_operation/stdout/hmc_hasenbusch_*active_mu*.out`

## 2026-06-05 追加：ratio force 符号修正与 stout reverse 复核

### 本轮修改

修改文件：

- `pyquda_core/pyquda/action/clover_hasenbusch.py`

备份文件：

- `pyquda_core/pyquda/action/clover_hasenbusch.py.codex_bak_ratio_force_sign_20260605_0539`

新增诊断脚本：

- `tests/hasenbusch_operation/scripts/smeared_force_check_12x32.py`
- `tests/hasenbusch_operation/scripts/stout_reverse_convention_check_12x32.py`

核心改动：`CloverWilsonHasenbuschTwistAction.force_v2()` 中 determinant-ratio 的 `pc_quda_norm` / `pc_quda_norm_chroma4` 聚合路径整体反号。原因是 `_add_sym_oddodd_quda_norm_deriv_force()` 的分支诊断显示它返回的是 `dS/dU` 方向导数口径，而当前 `HMC.loadGaugeMom()` 会把 `force_v2` 直接加到 momentum，因此 HMC 需要的是 `-dS/dU`。

### ratio-only smeared-link 层验证

修正前：

| run | eps | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---|---:|---:|---:|---:|---:|
| `smeared_cfg1000_ratio_only_active_mu_20260605` | 1e-4 | -1.629590988 | -1.632648733 | -3.262239721 | 1.99813 |
| 同上 | 3e-5 | -1.640090098 | -1.632648733 | -3.272738831 | 1.99546 |
| 同上 | 1e-5 | -1.554761548 | -1.632648733 | -3.187410281 | 1.95229 |

修正后：

| run | eps | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---|---:|---:|---:|---:|---:|
| `smeared_cfg1000_ratio_only_ratio_signfix_20260605` | 1e-4 | -1.633504871 | 1.632649007 | -0.000855864 | 0.000524 |
| 同上 | 3e-5 | -1.644253886 | 1.632649007 | -0.011604879 | 0.007058 |
| 同上 | 1e-5 | -1.624657307 | 1.632649007 | 0.007991700 | 0.004895 |

判断：ratio-only 的大符号错误已修正；剩余差异主要随 eps 波动，量级远小于修正前。

### full smeared-link 层验证

修正前 full smeared-link 层：

| eps | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---:|---:|---:|---:|---:|
| 1e-4 | -775.366718881 | 773.187848514 | -2.178870367 | 0.00281 |
| 3e-5 | -775.378663093 | 773.187848514 | -2.190814579 | 0.00283 |
| 1e-5 | 约 -775.259 | 773.187848514 | 约 -2.07 | 约 0.00267 |

修正后 full smeared-link 层：

| run | eps | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---|---:|---:|---:|---:|---:|
| `smeared_cfg1000_full_ratio_signfix_20260605` | 1e-4 | -775.366716553 | 775.361602480 | -0.005114072 | 6.60e-6 |
| 同上 | 3e-5 | -775.378655332 | 775.361602480 | -0.017052852 | 2.20e-5 |
| 同上 | 1e-5 | -775.259383954 | 775.361602480 | 0.102218526 | 1.32e-4 |

判断：ratio force 符号修正后，smeared-link 层 full force 已从约 `2.8e-3` 改善到 `1e-5` 到 `1e-4` 量级。说明 logdet/cancel/ratio/strange 的 smeared-link fermion force 组合已经基本闭合。

### thin-link stout reverse 后验证

同一组参数经过 `computeStoutForceQuda` 回传到 thin link 后：

| run | eps | 有限差分 | force 投影 | `FD+force` | 反号相对差 |
|---|---:|---:|---:|---:|---:|
| `force_cfg1000_full_ratio_signfix_20260605` | 1e-4 | -403.520727996 | 402.009761467 | -1.510966528 | 0.003744 |
| 同上 | 3e-5 | -403.517205268 | 402.009761467 | -1.507443801 | 0.003736 |
| 同上 | 1e-5 | -403.534783982 | 402.009761467 | -1.525022514 | 0.003779 |

判断：smeared-link 层已经闭合，但 thin-link 层仍有约 `3.7e-3` 的稳定残差。当前主疑点已经从 Hasenbusch ratio force 转移到 stout force reverse / thin-link 投影。

### stout reverse convention 诊断

新增脚本 `stout_reverse_convention_check_12x32.py` 测试了三种 `computeStoutForceQuda` 输入和两种输出解释：

- `raw_general_force`：直接传入 smeared-link general force。
- `smeared_left_force`：传入 `V_smeared * force`。
- `smeared_left_ah_traceless`：传入 `V_smeared * force` 的反厄米无迹部分。
- `project_Uthin_times_returned`：返回后按当前 HMC 口径做 `U_thin * returned` 投影。
- `direct_returned_ah_traceless`：把返回矩阵直接解释为反厄米无迹 momentum。

结果：

| input convention | output projection | force 投影 | `FD+force` | 反号相对差 |
|---|---|---:|---:|---:|
| raw_general_force | project_Uthin_times_returned | 402.009761506 | -1.519667371 | 0.003766 |
| raw_general_force | direct_returned_ah_traceless | -1561.574227660 | -1965.103656537 | 1.25841 |
| smeared_left_force | project_Uthin_times_returned | -255.877788564 | -659.407217441 | 1.63410 |
| smeared_left_force | direct_returned_ah_traceless | 416.481889535 | 12.952460658 | 0.03110 |
| smeared_left_ah_traceless | project_Uthin_times_returned | -279.143197562 | -682.672626439 | 1.69175 |
| smeared_left_ah_traceless | direct_returned_ah_traceless | 516.976870974 | 113.447442097 | 0.21944 |

判断：当前 PyQUDA 口径 `raw_general_force + U_thin 投影` 仍是六种组合里最好的，但它仍有约 `3.7e-3` 残差。因此问题不是简单的“多乘/少乘 link”接口误用，更可能在 QUDA stout reverse 链式公式、边界/ghost 处理、或 thin/smeared resident gauge 细节中。

### HMC smoke test 状态

尝试运行：

```bash
run_id=hmc_cfg1000_t1_stop2_ratio_signfix_20260605
trajectory_length=1.0
stop=2
placement=cancel_inner
integrator_inner=O4Nf5Ng0V
integrator_outer=O4Nf5Ng0V
nsteps_inner=2
nsteps_outer=7
ratio_force_mode=pc_quda_norm
cancel_force_mode=pc_quda_norm
matpc_type=odd_odd
cancel_constdet=yes
```

该 run 运行约 5 分钟仍未输出第一条 `Delta_E`，stdout 已超过 8500 行，主要被 Python force-term INFO 与 `performGaugeSmearQuda` 内部打印淹没。为避免继续占用 GPU 和交互通道，本次 smoke test 被终止，未作为 HMC 结论使用。

后续 HMC 验证前应先降低 Python force-term 日志，至少避免 `_add_pair_force_v2_quda_norm()` 每个分支每次 force 调用都输出 INFO。否则 `t=1` 的多时间尺度积分器会产生海量日志，严重拖慢调试和结果解析。

### 当前结论更新

1. 已确认并修复 determinant-ratio force 聚合路径的符号错误；这是一个实质性代码修复。
2. 修正后，smeared-link 层 full force 基本闭合，说明 Hasenbusch logdet/cancel/ratio/strange 的 smeared-link fermion force 组合已经不再是主要矛盾。
3. thin-link 层仍有约 `3.7e-3` residual，当前主疑点是 QUDA stout reverse 或 PyQUDA `HMC.loadGaugeMom()` 与 `computeStoutForceQuda` 的细节配合。
4. 已排除几种简单输入/输出 convention 错配；当前实现的 convention 仍是最好的，但不够好。
5. 尚未达到用户目标：还没有证明 `t=1, stop=1000` 能稳定运行，也没有证明 `Delta_E` 收敛。下一步应继续定位 stout reverse residual，并先加入安静日志模式再做 HMC smoke test。
## 2026-06-05 06:35 补充：quiet smoke 与纯 stout map 诊断


### 本轮新增/修改

- `pyquda_core/pyquda/action/clover_hasenbusch.py`
  - 新增 `self.force_debug_logging = False`，默认关闭 Hasenbusch pair-force 的逐项 INFO 日志。
  - 该改动不改变 force 数值路径；`force_term_history` 仍会记录诊断所需的 bilinear 项。
  - 备份：`pyquda_core/pyquda/action/clover_hasenbusch.py.codex_bak_quiet_force_log_20260605_0615`。
- `tests/hasenbusch_operation/scripts/stout_map_force_check_12x32.py`
  - 新增纯 stout 映射有限差分诊断脚本。
  - 目的：不经过 fermion action，直接测试 `V=stout(U)` 的链式导数和 `computeStoutForceQuda` 回传 force 的方向内积。

### 语法检查

```bash
python3 -m py_compile pyquda_core/pyquda/action/clover_hasenbusch.py
python3 -m py_compile tests/hasenbusch_operation/scripts/stout_map_force_check_12x32.py
```

结果：通过。

### 纯 stout map 诊断

运行：

```bash
mpiexec -n 4 --bind-to none python3 scripts/stout_map_force_check_12x32.py   --cfg 1000 --t-boundary -1   --run-id stoutmap_cfg1000_tbm1_20260605_0618   --eps 1e-4,3e-5,1e-5
```

输出：`csv/stout_map_force_check_stoutmap_cfg1000_tbm1_20260605_0618.csv`。

结果摘要：随机 general-link `sigma` 下，有限差分约 `-118.522957`，`computeStoutForceQuda` 回传后的 force 投影约 `-729.927455`，没有闭合。

解释：这个结果不能直接等价为物理 bug，因为随机 general-link `sigma` 未必满足 fermion force 输出的物理子空间/QUDA force kernel 输入约定。更可靠的既有证据仍是：fermion force 在 smeared-link 层已经闭合，而 thin-link 回传后剩约 `0.37%` 系统残差。因此该脚本当前主要作为后续测试不同 sigma 约定、投影约定和 stout reverse kernel 修改的诊断入口。

### t=1 stop=1 HMC smoke

共同参数：

- `trajectory_length=1.0`
- `t_boundary=-1`
- `placement=cancel_inner`
- `outer=O4Nf5Ng0V(7)`，`inner=O4Nf5Ng0V(2)`
- `hasenbusch_mu=0,0.0005,0.0015,0.0035,0.0075,0.0155`
- `ratio_force_mode=pc_quda_norm`
- `cancel_force_mode=pc_quda_norm`
- `matpc_type=odd_odd`
- `cancel_constdet=yes`

| run | cfg | stop | Delta_E | 接受情况 | plaquette 第一分量初值 | 末值 | 单步耗时 |
|---|---:|---:|---:|---|---:|---:|---:|
| `hmc_cfg1000_t1_stop1_quiet_20260605` | 1000 | 1 | -0.0981835206 | 1/1 accepted | 0.5355789712 | 0.5363697823 | 235.344 s |
| `hmc_cfg10001_t1_stop1_quiet_20260605` | 10001 | 1 | 0.2499912109 | 1/1 accepted | 0.5354183635 | 0.5354065195 | 239.974 s |

判断：quiet 日志修正后，两个目标组态都能完成 `t=1` 的第一条 trajectory，且这两条 `Delta_E` 未出现 NaN 或巨大爆炸。这是向目标前进的正信号，但仍不能证明 `stop=1000` 可稳定；当前每条 trajectory 约 4 分钟，继续直接长跑前仍应优先压低 full force 的剩余系统残差，并处理 `performGaugeSmearQuda` 每次调用输出未初始化 plaquette/qcharge 的噪声。

### 当前未完成项

- 尚未证明 `t=1, stop=1000` 在 cfg1000/cfg10001 上稳定。
- thin-link full force 仍有约 `0.37%` 到 `0.8%` 的有限差分残差，需要继续定位 stout reverse/边界/状态同步或 full monomial 组合项。
- `performGaugeSmearQuda` 在 HMC 中频繁打印 `step 000 plaquette`，且数值看起来是未初始化观测量；这不影响日志文件里的 HMC 结果，但会污染 stdout，建议下一步在 PyQUDA/QUDA 层关闭该输出或修正 obs 参数初始化。

## 2026-06-05 续测：thin-link 投影与 fat-force 边界探针

### 新增诊断脚本

- `tests/hasenbusch_operation/scripts/thin_projection_probe_12x32.py`

用途：固定 pseudofermion 和随机扰动方向后，对同一个 `computeStoutForceQuda` 返回的 thin-link general force，分别使用不同 thin gauge 口径投影成 momentum，再与 action finite-difference 比较。这里的“投影”指把 general-link force `F` 转成 HMC momentum 使用的反厄米无迹矩阵，当前 PyQUDA 口径为 `U_thin * F` 后取 anti-Hermitian traceless part。

### thin-link 投影口径结果

运行：

```bash
mpiexec -n 4 --bind-to none python3 scripts/thin_projection_probe_12x32.py \
  --cfg 1000 --t-boundary -1 \
  --run-id thinproj_cfg1000_tbm1_ratio_signfix_20260605 \
  --eps 1e-4,3e-5,1e-5 \
  --ratio-force-mode pc_quda_norm \
  --cancel-force-mode pc_quda_norm \
  --cancel-constdet yes --matpc-type odd_odd
```

输出：`csv/thin_projection_probe_thinproj_cfg1000_tbm1_ratio_signfix_20260605.csv`。

| projection | eps | force 投影 | 反号相对误差 |
|---|---:|---:|---:|
| `physical_thin_U` | 1e-4 | 272.761477691 | 0.324049 |
| `resident_ap_thin_U` | 1e-4 | 402.009761506 | 0.003749 |
| `hmc_current_path` | 1e-4 | 402.009761506 | 0.003749 |
| `physical_thin_U` | 3e-5 | 272.761477691 | 0.324061 |
| `resident_ap_thin_U` | 3e-5 | 402.009761506 | 0.003766 |
| `hmc_current_path` | 3e-5 | 402.009761506 | 0.003766 |

判断：当前 HMC 使用的 `resident_ap_thin_U` 投影就是三者中最优，并且与 `hmc_current_path` 完全一致；使用 physical thin link 投影反而把误差放大到约 `32%`。因此，thin-link 层约 `0.37%` residual 不是 PyQUDA 投影时少乘/多乘 anti-periodic T 边界造成的。

### fat-force anti-periodic T 边界探针

Chroma `fatForceToThin()` 在进入 `deriv_recurse()` 前会对 fat force 执行一次 fermion 边界条件 `modify(F_thin)`。为测试它在 PyQUDA/QUDA 中是否可近似为 “对 smeared-link force 做 T 方向边界翻号”，新增 `apply_ap_t` 探针：在调用 `computeStoutForceQuda` 前对 `force_v2` 执行一次 `setAntiPeriodicT()`。

运行：

```bash
mpiexec -n 4 --bind-to none python3 scripts/thin_projection_probe_12x32.py \
  --cfg 1000 --t-boundary -1 \
  --run-id thinproj_cfg1000_tbm1_fatbc_probe3_20260605 \
  --eps 3e-5 \
  --ratio-force-mode pc_quda_norm \
  --cancel-force-mode pc_quda_norm \
  --cancel-constdet yes --matpc-type odd_odd
```

输出：`csv/thin_projection_probe_thinproj_cfg1000_tbm1_fatbc_probe3_20260605.csv`。

| projection | eps | force 投影 | 反号相对误差 |
|---|---:|---:|---:|
| `raw:physical_thin_U` | 3e-5 | 272.761477686 | 0.324061 |
| `raw:resident_ap_thin_U` | 3e-5 | 402.009761501 | 0.003766 |
| `apply_ap_t:physical_thin_U` | 3e-5 | 324.212835932 | 0.196557 |
| `apply_ap_t:resident_ap_thin_U` | 3e-5 | 299.990887057 | 0.256582 |

判断：简单对 fat force 做 anti-periodic T 翻号不是正确修复；它会把 residual 从约 `0.00377` 明显变坏到 `0.20-0.26`。因此 Chroma 的 `fbc->modify(F_thin)` 不能直接照搬为 PyQUDA `force_v2.setAntiPeriodicT()`。

### 当前结论修正

已排除两个 PyQUDA 侧简单解释：

1. `computeStoutForceQuda` 返回 force 后的 thin-link 投影口径并未选错；当前 `resident_ap_thin_U` 是最优口径。
2. fat-force 进入 stout reverse 前简单施加 anti-periodic T 翻号不是修复方向。

剩余主疑点继续集中在 QUDA stout reverse 内部实现或更细的输入 force 约定上，包括：

- `include/kernels/gauge_stout_force.cuh` 中 `STOUTForceLambda` / `STOUTForceSigma` 对 Chroma `deriv_recurse()` 的实现差异。
- `lambda.exchangeExtendedGhost(lambda.R())` 与负向 staple 分支在多 GPU T 切分下的边界/ghost 数据。
- QUDA `performGaugeSmearQuda` 无条件打印未初始化 observable 的噪声问题；这是调试效率问题，不是当前 residual 的直接证据。

当前仍不建议直接启动 `t=1, stop=1000`；应先把 `cfg1000/cfg10001` 的 `t=1, stop=5/10` quiet 版本补齐，并继续把 thin-link force residual 压低或确认它不会破坏 HMC 接受率。

## 2026-06-05 07:30 补充：t=1 stop=5 双组态 quiet 验证

本轮继续使用头文件参数口径：

- `tau0=1.0`，对应 runner 的 `trajectory_length=1.0`。
- `boundary=1 1 1 -1`，对应 `t_boundary=-1`。
- `rho=0.125`，`u0=0.855453`，`v0=0.951479`。
- outer/inner 积分器步数分别为 `7` 和 `2`。
- light Hasenbusch shift 链为 `0,0.0005,0.0015,0.0035,0.0075,0.0155`，等价于头文件中 light mass 链从 `-0.2770` 到 `-0.2615`。

共同 HMC 参数：

- `placement=cancel_inner`
- `integrator_outer=O4Nf5Ng0V(7)`
- `integrator_inner=O4Nf5Ng0V(2)`
- `ratio_force_mode=pc_quda_norm`
- `cancel_force_mode=pc_quda_norm`
- `matpc_type=odd_odd`
- `cancel_constdet=yes`
- `seed=10086`

指标含义：

- `plaquette 第一分量`：日志中 `Plaquette = [a,b,c]` 的第一个数 `a`。
- `plaquette 波动范围`：本 run 内 `max(a)-min(a)`。
- `plaquette 首末变化`：最后一条记录的 `a` 减去初始 `a`。
- `Delta_E 最大绝对值`：所有 trajectory 中 `max(abs(Delta_E))`，用于观察能量守恒误差的最坏情况。

| run | cfg | stop | 完成情况 | 接受率 | Delta_E 列表 | Delta_E 平均值 | Delta_E 最大绝对值 | plaquette 第一分量初值 | 末值 | 波动范围 | 首末变化 |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| `hmc_cfg1000_t1_stop5_quiet_ratiofix_20260605` | 1000 | 5 | ok | 3/5 = 0.60 | `[-0.101074, 0.167723, 0.256076, -0.215377, 0.268020]` | 0.075073 | 0.268020 | 0.535578971 | 0.537380081 | 0.001801110 | +0.001801110 |
| `hmc_cfg10001_t1_stop5_quiet_ratiofix_20260605` | 10001 | 5 | ok | 4/5 = 0.80 | `[0.249482, -0.771756, 0.811423, 0.548608, 0.345706]` | 0.236693 | 0.811423 | 0.535418364 | 0.534882563 | 0.000964890 | -0.000535801 |

判断：

1. 两个目标组态都已经能在 `t=1`、头文件步数 `outer=7/inner=2` 下完成 `stop=5`，没有 NaN 或 QUDA abort。这比早期 `t=1` 直接炸掉是实质进展。
2. `cfg1000` 的短跑能量误差较温和，`Delta_E 最大绝对值` 约 `0.268`。
3. `cfg10001` 的短跑能量误差更大，`Delta_E 最大绝对值` 约 `0.811`，但 5 步内接受率仍为 `0.80`。
4. 这仍不足以证明目标完成；`stop=5` 只能作为 smoke test。当前已启动下一阶段 `cfg1000 t=1 stop=20`，用于检查接受率和 `Delta_E` 是否随 trajectory 增加变坏。

新增远端作业：

```bash
run_id=hmc_cfg1000_t1_stop20_quiet_ratiofix_20260605
cfg=1000
trajectory_length=1.0
stop=20
outer=O4Nf5Ng0V(7)
inner=O4Nf5Ng0V(2)
```

截至本节写入时，该 `stop=20` 作业正在 p100 后台运行，PID 为 `2365038`。

## 2026-06-05 08:10 补充：cfg1000 t=1 stop=20 完成

`hmc_cfg1000_t1_stop20_quiet_ratiofix_20260605` 已完成，运行状态为 `ok`，无 NaN、无 QUDA abort。

| run | cfg | stop | 完成情况 | 接受率 | Delta_E 平均值 | Delta_E 最小值 | Delta_E 最大值 | Delta_E 最大绝对值 | plaquette 第一分量初值 | 末值 | 波动范围 | 首末变化 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmc_cfg1000_t1_stop20_quiet_ratiofix_20260605` | 1000 | 20 | ok | 13/20 = 0.65 | 0.142092 | -0.578950 | 0.800359 | 0.800359 | 0.535578971 | 0.537090491 | 0.001862877 | +0.001511520 |

`Delta_E` 序列：

```text
-0.0980301, 0.163797, 0.260785, -0.215044, 0.261894,
-0.108945, 0.307090, 0.253178, 0.800359, 0.431216,
0.214432, 0.0659754, 0.702247, 0.0133662, -0.578950,
0.448354, -0.262009, 0.243621, 0.360876, -0.422369
```

判断：

1. `cfg1000` 在 `t=1`、头文件步数 `outer=7/inner=2` 下已经从 `stop=5` 扩展到 `stop=20`，仍能完整跑完。
2. 接受率 `0.65` 可用，但不算非常宽裕；`Delta_E 最大绝对值` 从 stop=5 的约 `0.268` 扩大到约 `0.800`，说明长一些后会见到更大的能量误差尾部。
3. plaquette 第一分量在 20 步内没有单调崩坏，波动范围约 `0.00186`；不过 20 步仍不足以证明最终目标 `stop=1000` 稳定。
4. 已启动 `cfg10001 t=1 stop=20` 同参数对照，PID 为 `2393465`。待它完成后再判断是否进入更长 run 或先做积分器步数对比。

新增远端作业：

```bash
run_id=hmc_cfg10001_t1_stop20_quiet_ratiofix_20260605
cfg=10001
trajectory_length=1.0
stop=20
outer=O4Nf5Ng0V(7)
inner=O4Nf5Ng0V(2)
```

截至本节写入时，该 `cfg10001 stop=20` 作业正在 p100 后台运行，PID 为 `2393465`。

## 2026-06-05 08:14 补充：长程启动脚本整理

为避免误用早期脚本，新增了当前验证口径对应的长程启动脚本：

```text
/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/hasenbusch_operation/scripts/run_hasenbusch_ratiofix_long.sh
```

该脚本固定使用当前短跑验证中的参数组合：

- `t_boundary=-1`
- `trajectory_length=1.0`
- `seed=10086`
- `nsteps_outer=7`
- `nsteps_inner=2`
- `placement=cancel_inner`
- `integrator_outer=O4Nf5Ng0V`
- `integrator_inner=O4Nf5Ng0V`
- `hasenbusch_mu=0,0.0005,0.0015,0.0035,0.0075,0.0155`
- `ratio_force_mode=pc_quda_norm`
- `cancel_force_mode=pc_quda_norm`
- `matpc_type=odd_odd`
- `cancel_constdet=yes`
- `mpiexec -n 4 --bind-to none`

用法示例：

```bash
STOP=100 CFG_LIST="1000 10001" RUN_SUFFIX=ratiofix_probe bash scripts/run_hasenbusch_ratiofix_long.sh
STOP=50 CFG_LIST="10001" NSTEPS_OUTER=14 NSTEPS_INNER=4 RUN_SUFFIX=ratiofix_o14_i4 bash scripts/run_hasenbusch_ratiofix_long.sh
```

可覆盖的环境变量：

- `STOP`：trajectory 数量，默认 `1000`。
- `CFG_LIST`：要跑的组态列表，默认 `1000 10001`。
- `NSTEPS_OUTER` / `NSTEPS_INNER`：外层/内层积分器步数，默认 `7/2`。
- `INTEGRATOR_OUTER` / `INTEGRATOR_INNER`：外层/内层积分器类型，默认 `O4Nf5Ng0V/O4Nf5Ng0V`。
- `RUN_SUFFIX`：run_id 后缀，便于区分积分器扫描。

注意：旧脚本 `scripts/run_full_hasenbusch_stop1000.sh` 是早期版本，仍使用 runner 默认 force 参数和 `mpiexec -n 1`，不能用来代表当前已经通过短跑验证的 Hasenbusch ratiofix 口径。

解析注意事项：`scripts/parse_hmc_logs.py` 会把日志中 `Delta_E` 数量少于 `stop` 且尚无 `Run status = ok` 的 run 标为 failed/incomplete。因此它适合在 run 完成后生成正式 CSV；若对正在运行的日志调用，只能把 `n_delta_e`、接受/拒绝计数当作进度信息，不能把 temporary failed 当成物理失败结论。

## 2026-06-05 08:53 补充：cfg10001 t=1 stop=20 完成

`hmc_cfg10001_t1_stop20_quiet_ratiofix_20260605` 已完成，运行状态为 `ok`，无 NaN、无 QUDA abort。

同一参数组合下两个目标组态的 stop=20 对照如下：

| run | cfg | stop | 完成情况 | 接受率 | Delta_E 平均值 | Delta_E 最小值 | Delta_E 最大绝对值 | plaquette 第一分量初值 | 末值 | 平均值 | 波动范围 | 首末变化 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmc_cfg1000_t1_stop20_quiet_ratiofix_20260605` | 1000 | 20 | ok | 13/20 = 0.65 | 0.142092 | -0.578950 | 0.800359 | 0.535578971 | 0.537090491 | 0.536730846 | 0.001862877 | +0.001511520 |
| `hmc_cfg10001_t1_stop20_quiet_ratiofix_20260605` | 10001 | 20 | ok | 13/20 = 0.65 | 0.254891 | -0.785117 | 1.493381 | 0.535418364 | 0.535642804 | 0.535394693 | 0.001231824 | +0.000224441 |

`cfg10001` 的 `Delta_E` 序列：

```text
0.249103, -0.777215, 0.814025, 0.547272, 0.338496,
0.070546, 1.088324, 0.439707, -0.785117, 0.512608,
-0.053770, 0.035195, -0.065815, 0.730877, 0.438489,
-0.248148, 1.493381, 0.525577, 0.089263, -0.344976
```

判断：

1. 两个目标组态现在都能在 `t=1`、头文件步数 `outer=7/inner=2`、`stop=20` 下完整跑完。这说明 Hasenbusch/logdet/cancel/ratio-chain 以及 stout reverse 的主要数据流已经从“直接崩溃/巨大能量误差”推进到可运行状态。
2. `cfg10001` 的 plaquette 第一分量没有表现为单调崩坏；20 步内波动范围约 `0.00123`，首末变化约 `+0.000224`。
3. 但 `cfg10001` 的能量误差尾部仍偏大，`Delta_E 最大绝对值` 达到约 `1.49`，接受率只有 `0.65`。这不足以直接证明 `stop=1000` 会稳定，尤其长跑中可能继续遇到更大的能量误差尾部。
4. 因此下一步不直接上 `stop=1000`，而是先做积分步数加密对照，检验当前 `O4Nf5Ng0V(7)/O4Nf5Ng0V(2)` 是否步长偏粗。

已启动新的积分器步数对照：

```text
run: hmc_cfg10001_t1_stop20_ratiofix_o14_i4_20260605_085625
cfg: 10001
t_boundary: -1
trajectory_length: 1.0
stop: 20
outer: O4Nf5Ng0V(14)
inner: O4Nf5Ng0V(4)
ratio_force_mode: pc_quda_norm
cancel_force_mode: pc_quda_norm
matpc_type: odd_odd
cancel_constdet: yes
mpiexec pid: 2430232
```

备注：第一次 driver `driver_ratiofix_o14_i4_20260605_085248.out` 在进入 MPI 前退出，原因是脚本早期版本在 `source set_PyQUDA_env.sh` 前启用了 `set -u`，而环境脚本引用未定义的 `PYTHONPATH`。这不是 HMC 物理失败。脚本已修正为 source 环境后再启用 `set -u`，并以 `085625` run 重新启动。

预期判据：如果 `outer=14/inner=4` 显著降低 `Delta_E 最大绝对值` 并提高接受率，则说明当前主要瓶颈更偏积分步长；若加密后改善不明显，则需要继续回到 force normalization / stout reverse 残差约 `3.7e-3` 的问题上。

## 2026-06-05 11:03 补充：cfg10001 积分器加密对照完成

`hmc_cfg10001_t1_stop20_ratiofix_o14_i4_20260605_085625` 已完成，运行状态为 `ok`，无 NaN、无 QUDA abort。参数为：

```text
cfg=10001
t_boundary=-1
trajectory_length=1.0
stop=20
outer=O4Nf5Ng0V(14)
inner=O4Nf5Ng0V(4)
placement=cancel_inner
ratio_force_mode=pc_quda_norm
cancel_force_mode=pc_quda_norm
matpc_type=odd_odd
cancel_constdet=yes
```

同一组态 `cfg10001` 的原步数和加密步数对比如下：

| run | outer / inner | 完成情况 | 接受率 | Delta_E 平均值 | Delta_E 最小值 | Delta_E 最大绝对值 | plaquette 第一分量初值 | 末值 | 平均值 | 波动范围 | 首末变化 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hmc_cfg10001_t1_stop20_quiet_ratiofix_20260605` | 7 / 2 | ok | 13/20 = 0.65 | 0.254891 | -0.785117 | 1.493381 | 0.535418364 | 0.535642804 | 0.535394693 | 0.001231824 | +0.000224441 |
| `hmc_cfg10001_t1_stop20_ratiofix_o14_i4_20260605_085625` | 14 / 4 | ok | 15/20 = 0.75 | 0.106648 | -0.630377 | 0.630377 | 0.535418364 | 0.535483519 | 0.535571071 | 0.001712974 | +0.000065155 |

加密步数 run 的 `Delta_E` 序列：

```text
0.323828, -0.129669, 0.120041, 0.333724, 0.152001,
0.075138, 0.572128, 0.453110, 0.111021, 0.280852,
0.282835, -0.124629, -0.053897, 0.074232, 0.205996,
-0.202098, 0.297573, 0.137800, -0.146649, -0.630377
```

判断：

1. 加密积分步数显著降低了 `Delta_E` 尾部：最大绝对值从 `1.493381` 降到 `0.630377`，接受率从 `0.65` 提高到 `0.75`。
2. `Delta_E` 平均值从 `0.254891` 降到 `0.106648`，说明改善不是单个异常点消失，而是整体能量守恒变好。
3. plaquette 第一分量没有单调漂移，首末变化只有 `+0.000065155`；不过 20 步样本仍太短，不能证明 `stop=1000` 下也稳定。
4. 加密步数的 plaquette 波动范围 `0.001712974` 略大于原步数 `0.001231824`，但这主要是短样本波动，不能单独解释为稳定性变差。
5. 当前证据支持“积分步长偏粗是主要瓶颈之一”。但最后一步 `Delta_E=-0.630377` 仍是全程最大绝对值，说明仍应继续查 full force/stout reverse 剩余残差，不能直接跳到 `stop=1000`。

下一步建议：

1. 先跑 `cfg10001, t=1, outer=14/inner=4, stop=100`，确认能量误差尾部是否继续受控。
2. 补一个状态不变性诊断：验证 `samplePhi()`、`fermionAction()` 和零 `force_v2` 的 `loadGaugeMom()` 不会意外改变 momentum。当前 `hmc.py` 依赖 `force_v2` 被清零，逻辑能跑，但状态耦合偏脆。
3. 如果 `stop=100` 仍稳定，再考虑 `cfg1000/cfg10001` 的 `stop=200/500`，最后再上 `stop=1000`。
