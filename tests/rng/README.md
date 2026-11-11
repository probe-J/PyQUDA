# Chroma 随机数与 HMC 对齐的最小验证工具

本目录从实验快照 `25d5e347a6532fe2ba5111bad6c7e204f0a9ea1b` 精选重建，
初次整理提交为 `9e2d421`，基线为 `b808cbdfa8ab607ae565fb0705b733a3126df917`。
本次继续提取 2026-11-11 CG/MG 对拍所需的真实质量比、头文件配置与求解管理；
RNG 核心和 adapter 无需再修改。默认 PyQUDA 原生 RNG 不变；
只有显式挂载 `chroma_rng_adapter.attach_chroma_rng()` 才启用 Chroma 流。
这里是实验验证工具，不是已经完成跨平台数值认证的生产配置。

## 保留了什么

| 类别 | 文件或行为 | 保留理由 |
| --- | --- | --- |
| RNG 核心 | `pyquda_core/pyquda/chroma_rng.py` | 47-bit LCG、float32 均匀数、高斯变换、坐标映射、动量打包 |
| HMC 对拍依赖 | `action/clover_hasenbusch.py` | 全局体积常数、Chroma logdet 零点、constdet cancel、双线性力符号和归一化 |
| 奇夸克对拍依赖 | `action/clover_wilson.py` | 显式预条件奇偶性、action rational、symmetric odd-odd 力 |
| 构造接口 | `hasenbusch.py` | Chroma shift 到 QUDA `mu/kappa` 的映射 |
| 注入器 | `chroma_rng_adapter.py` | 同一条流生成动量和全部 eta，嵌套 HMC 共享流，显式接受判定开关 |
| 复现支持 | runner、配置、XML、dump 工具与 CPU 测试 | 用最小依赖重做验证，不依赖旧实验目录 |

action/force 修改不是 RNG 算法本身，而是原实验用于对齐 HMC 的数值依赖。
这些改动会改变哈密顿量零点或力，不能只凭 RNG 单元测试认定其数学正确。
沿用已有 `STSForceGrad`，不引入诊断用 `STSLeapfrog`。
`hmc.py` 增加嵌套 MG 生命周期与 FG 异常恢复；求解管理仅提取此次对拍所需部分，
不包含 predictor 优化、长期性能记录或旧实验扫描系统。

## 初次清理已有修正

- 挂载时立即替换 `sampleEta`，不再用初始空缓存决定是否替换。
- 高斯计算按显式 `precision` 分支；均匀数始终是 float32，不能据其 dtype 推断高斯计算精度。
- Chroma 接受规则为 `DeltaH <= 0` 直接接受且不抽数；正值时抽一次 uniform。
  对照源码：Chroma `lib/update/molecdyn/hmc/global_metropolis_accrej.cc`。
- `detach()` 恢复原实例绑定；删除对后续 solver 生命周期接口的依赖。
- 修复无效的 `or True` 自检，补全状态推进、精度和场布局测试。
- dump 比较必须覆盖完整坐标及分量集合；缺格点、重复、非有限值均失败。
- 排除新增的 shift 导数诊断旋钮，以及每次力计算强制打印场范数的代码。
- 保留 pair-force 原有 `force_sign/force_coeff_scale` 倍率，避免接受参数却静默失效。
- 修复 asymmetric action-rational 漏扣 inactive clover determinant；symmetric
  odd-odd 路径仍为显式 const-det，完整 determinant 的 logdet 由调用方另行构造。
- symmetric 力只对本 action 的增量应用 `dt`；临时场、dagger 与 resident 标志在异常时恢复。
  尚未实现的 symmetric even-even 和 symmetric 直接动量更新接口明确抛出异常。

兼容性注意：`CloverHasenbuschConfig.mu_scale=None` 表示输入 Chroma shifts，
自动乘 `1/kappa`。基线配置若已把 shifts 换算为 QUDA 单位，迁移时必须显式传
`mu_scale=1.0`，不能再乘一次；示例配置把输入 shifts、倍率和实际 QUDA shifts 都记录下来。

## CPU 验证

从仓库根目录执行，Python 环境只需 NumPy：

```bash
python3 -m unittest discover -s tests/rng -v
python3 pyquda_core/pyquda/chroma_rng.py
```

测试包含固定 seed 的 float32 位比较、独立整数 LCG 公式、状态保存恢复、
single/double 高斯分支、eta 缩放与奇偶映射、TA 投影、QUDA 10-real 动量打包、
模拟不同 rank 切分后的坐标一致性，以及注入器每轨迹刷新、接受判定和恢复。
固定小向量位于测试源码中；这是算法回归数据，**不是本次新生成的外部 Chroma golden dump**。
假对象测试不等于真实 MPI 或 GPU 内核测试。

## 字段对拍

`chroma_rng_dump.cc` 要用目标 Chroma/QDP++ 的编译与链接参数构建。
实际精度由该 build 决定，命令行 precision 只记录元数据，不改变计算精度。
单独运行 `uniforms`、`eta`、`mom` 时，两端都从同一 seed 重新开始：

```bash
chroma_rng_dump 4 4 4 4 12345 uniforms results/chroma_uniforms.txt double
python3 -c 'from pyquda.chroma_rng import ChromaRNG; ChromaRNG((4,4,4,4)).dump_reference("results/pyquda_uniforms.txt",12345,"uniforms")'
python3 tests/rng/compare_dump.py results/chroma_uniforms.txt.rank0 results/pyquda_uniforms.txt
```

先创建独立输出目录。多 rank 的 Chroma 文件应合并后比较，不能只拿 rank0 与全格场比较。
uniforms 按 float32 位相等；double eta/mom 使用
`abs(a-b)/max(1,abs(a)) <= 1e-12`；single Gaussian 使用历史数值容差 `2e-6`，
不是 single Gaussian 位级相等的声明。eta dump 未乘采样所需的 `sqrt(0.5)`；
mom dump 已乘 `sqrt(0.5)` 并做 traceless anti-Hermitian 投影。
PyQUDA 注入动量使用其自身归一化，不可把两种 dump 直接混用。

## HMC 验证边界

本节的 `run_hmc.py` 和 `chroma_12x32.xml` 是历史 twist 诊断入口；新对拍应使用下节
`run_header_hmc.py`，不能将两个入口的 action 集合或成功记录混用。

`run_hmc.py --help` 提供本机可检查的入口。运行需要安装匹配的 PyQUDA/QUDA 扩展、
CUDA/MPI、CuPy，以及外部 LIME 组态；本仓库不携带组态、编译产物或历史实验数据。
使用同一配置分别运行 native/chroma 是方案 A；与 XML 的 Chroma 输出配对是方案 B。
两者必须记录采样顺序、接受判定 RNG 开关、预条件形式、action/force 集合和积分器。

XML 保留历史 Chroma `tau0=0.1`，而历史归一化映射采用 PyQUDA `t=tau0/sqrt(2)`。
这不是“两个参数都直接设 0.1”的对照；runner 记录两个值，禁止隐藏换算。
XML 是历史参考模板，不代表已和清理后的 runner 重新完成跨平台对拍。
源代码所依赖的 QUDA 私有接口及 Chroma/QDP build 版本也必须单独锁定。
历史 XML 的 `MaxCG=5000` 与最小 PyQUDA 配置的 `maxiter=1000` 不同；
比较前须明确统一求解预算或逐次核实收敛，不能据此模板自动宣称全部条件一致。
该历史 XML 的 strange 为 const-det，未独立列出 strange 质量的 logdet；最小 runner
为复现它保留同一 action 集合。这不能自动视为完整的物理“两轻一奇”生产配置。
如需完整 determinant，应两端同时补齐 strange logdet 并重新验证，不在清理提交时单边改变。

例如在具有匹配扩展的 GPU 环境中，从仓库根目录运行（两个输出目录均应不存在）：

```bash
mpiexec -n 4 python3 tests/rng/run_hmc.py --gauge /path/to/cfg1000.lime --output-dir /path/to/runs/native --arm native --ntraj 2
mpiexec -n 4 python3 tests/rng/run_hmc.py --gauge /path/to/cfg1000.lime --output-dir /path/to/runs/chroma --arm chroma --ntraj 2 --use-chroma-accept
```

只检查动量注入时增加 `--mom-action-only`；该模式不跑费米子求解或 MD。
如仅隔离场生成机制而不比较接受判定流，Chroma 臂不要加 `--use-chroma-accept`，
并在结果中明确这一限制。输出包含逐 rank 场指纹、流状态、哈密顿量及接受判定；
组态检查点不是完整 RNG 状态重启文件，不能拿来保证位级续跑。

## 新的按头文件构造入口

以下为本次从成功实验提取的必要代码，已归并为可复用入口：

| 文件 | 作用 |
| --- | --- |
| `action/clover_mass_ratio.py` | 真实 bare mass 的 Schur 算子质量比，四项导数，dt 只乘一次 |
| `hasenbusch.build_clover_hmc_from_monomials` | 从声明构造 gauge、light/strange logdet、ratio、cancel、strange；显式层级和采样顺序 |
| `solver_accel.py` | 采样/能量双精度 CG，MD 可选 MG，原正规方程残差、FG 冻结与资源清理 |
| `extract_chroma_header.py` | 从 LIME 生产头生成 JSON、Chroma input XML 和审计记录 |
| `run_header_hmc.py`、`chroma_hmc_support.py` | 统一 CG/MG 入口、QIO 校验与列向量重酉化、固定场导数检查 |
| `compare_hmc_results.py` | 完整轨迹数、DeltaH、正式 plaquette、接受状态和 RNG 末态配对比较 |

真实标量质量变化不能用 twist shift 替代。`copyList` 的 H/MD 名称解析为同一
物理 action，保留各自求解容差，禁止重复采样或遗漏。light 和 strange 的 logdet
均单独构造，rational 系数从输入复制。strange 各极点使用同一残差目标；
`CloverWilsonAction` 原生默认的 residue 容差策略不变。

QIO 读取检查尺寸、精度和 SciDAC checksum，复现 Chroma 列向量 Gram–Schmidt，
避免不同 SU(3) 投影改变初始场。注入动量为 `P_QUDA=sqrt(2)*P_Chroma`，
两种 RNG 臂均显式用 `t_QUDA=tau0_Chroma/sqrt(2)`，输出同时记录两值。
所有 fixed-field 检查后会释放各层 MG 基、沿用原配置冷启动，并重置组态和物理 RNG，
再开始正式轨迹，避免检查阶段的求解历史改变正式对照。

输入只读，输出目录须在源码树外且事先不存在。未知 monomial、twist、各向异性、
非两层 STSForceGrad 等输入会明确拒绝，不根据文件名推测参数。

```bash
python3 -B tests/rng/extract_chroma_header.py \
  --gauge /data/cfg.lime --output-dir /runs/header --ntraj 2
mpiexec -n 4 python3 -B tests/rng/run_header_hmc.py \
  --config /runs/header/config.json --output-dir /runs/pyquda-cg \
  --solver cg --arm chroma --grid 1 1 2 2 --ntraj 2
```

提高 cancel 迭代上限时，可在转换时显式传 `--cancel-maxiter 5000`，该覆盖同时进入
XML/JSON 和审计记录。方案 A 使用 `--arm native`、`--arm chroma` 两个独立输出目录；
方案 B 使用生成的 Chroma XML 与 Chroma RNG 臂配对。

### MG 配置与验证

复制 `solver_mg.example.json` 到运行目录并按目标 build 调整。默认 `seed_backend=api`
要求实际加载的 QUDA 导出 `hmc_quda_set_mg_setup_seed(uint64_t)`。若 build 仅有已核实的
固定 1234 初始化，可显式选 `builtin_fixed1234`，同时给出 `setup_seed=1234`、实际加载库
绝对 `library_path` 和 `library_sha256`。运行时检查唯一映射、设备/inode 与哈希；不匹配
即报错。这是既有 build 的显式证明，不是任意设 seed 的替代 API。

```bash
mpiexec -n 4 python3 -B tests/rng/run_header_hmc.py \
  --config /runs/header/config.json --output-dir /runs/pyquda-mg \
  --solver mg --solver-config /runs/solver_mg.json \
  --arm chroma --grid 1 1 1 4 --ntraj 2
```

MG 只用于 ratio/cancel 的 MD，采样和能量仍用 CG。先解伴随与原方程，再复核原正规
方程残差；证据按调用顺序逐一检查。普通 MD 允许显式记录的重建重试，但必须最终
收敛到该 action 的 MD 容差；FG 临时场只能借用已有基，禁止失败、refresh/rebuild
或静默 fallback。未闭合的失败调用不能由另一调用的成功掩盖。
进程网格须满足格点整除与聚合块约束，禁止 QUDA 静默缩小块。例如 12³×32、block=2⁴
可用 1×1×1×4 网格。输出仅包含必要求解证据；`--mode gate` 不跑正式轨迹。

### 逐轨迹比较

提供按轨迹顺序排列的 Chroma restart，以校验 RNG 状态连续性与末态；
正式 plaquette 指接受/拒绝后保留组态的值，proposal 单独记录。

```bash
python3 -B tests/rng/compare_hmc_results.py \
  --chroma /runs/chroma/output.xml --pyquda /runs/pyquda-cg/rank0.jsonl \
  --chroma-restart /runs/chroma/restart_1.xml \
  --chroma-restart /runs/chroma/restart_2.xml \
  --ntraj 2 --output /runs/comparison.json
```

默认要求 DeltaH 绝对差 ≤1e-7、plaquette 绝对差 ≤1e-12、接受状态和 seed 全等；
缺轨迹、非有限值和 seed 断链均失败。可显式改阈值，输出保留所用容差。

## 本次未纳入

实验 CSV/JSON、日志、restart/out XML、日期目录、重复参数扫描、一次性 patch/revert、
Apple 元数据、组态与大 dump、predictor 优化和长期计时 histories 不进入新提交。
原 RNG 分支和原工作区实物不删除；新分支从父节点重建，并非在污染提交之后删除文件。
因此新分支历史不含污染提交，但保留原分支仍会保留原仓库历史占用。

## 验证状态

CPU 单元测试与静态语法检查在本次整理时重新执行，详细覆盖见上述测试。
初次整理为 72 项；2026-11-12 本次新增代码的完整 CPU 测试共 177 项通过。
新增覆盖质量比、声明构造、头参数检查、CG/MG 残差、FG 状态与异常恢复。
原始外部 Chroma 字段 dump 未在本次已检查的源快照中找到，未以自行生成数据冒充它。
本次未编译 Chroma dump 工具；真实 cfg10001 头文件已完成转换及配置检查。
本次在 p100 的隔离源码覆盖层重新运行 12³×32 cfg10001，CG/MG 各两条轨迹完成。
两种模式均通过全部 10 个 action 的固定场力检查；每条 MG 轨迹还通过求解证据检查。
与各自 Chroma 参考配对的结果如下，数值为两条轨迹中绝对差的最大值：

| PyQUDA 模式 | DeltaH 最大绝对差 | plaquette 最大绝对差 | 接受状态 / 每轨 RNG 末态 |
| --- | --- | --- | --- |
| CG | 1.278e-8 | 1.499e-14 | 两条均一致 |
| MD 使用 MG，采样/能量使用 CG | 5.958e-8 | 3.440e-13 | 两条均一致 |

两侧初始 seed 也核验一致，四条 PyQUDA 轨迹全部接受。两模式第 1/2 条末态 seed
分别为 70440947488745 / 84623469229929；动量和 eta 均在第 2 条重新生成。
实测参数为 4D stout、rho=0.125、一层、反周期时间边界、两层 STSForceGrad、
outer/inner=7/2，Chroma tau0=1、PyQUDA t=1/sqrt(2)，其余作用量参数由头文件导入。
cancel 的头文件上限 200 不足，显式覆盖为 5000 并保留审计；ratio 上限 1000，
strange 上限 5000。CG 用 1×1×2×2 网格，MG 用 1×1×1×4 和 2⁴ 聚合块。
MG setup seed 使用已加载库的固定 1234 证明；仓库模板使用通用 seed API，
不把机器路径或动态库哈希写入模板。

Chroma 对照使用 2026-11-11 已有正式 XML/restart，本次未重新运行 Chroma。
输出只留在 p100 的 `test_stout_smesr/codex_operation/rng_clean_validation_20261112`；
复现入口和比较命令见上文，输入配置及机器 MG 配置随运行目录保留，不提交实验产物。
验证不替换 p100 当前安装或工作区，真实轨迹对拍也不等于热平衡物理量认证。
发布的是可审查、可继续验证的干净功能分支，不是新的完整 A/B 等价性通过报告。
