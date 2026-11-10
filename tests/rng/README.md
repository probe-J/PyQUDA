# Chroma 随机数机制的最小验证工具

本目录从实验快照 `25d5e347a6532fe2ba5111bad6c7e204f0a9ea1b` 精选重建，
基线为 `b808cbdfa8ab607ae565fb0705b733a3126df917`。默认 PyQUDA 原生 RNG 不变；
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
`hmc.py` 未改；沿用基线已有 `STSForceGrad`，不引入诊断用 `STSLeapfrog`。
本次不重写既有实验 action API，不把后续 solver 加速合并进来。

## 本次额外修正

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

## 本次未纳入

实验 CSV/JSON、日志、restart/out XML、日期目录、重复参数扫描、一次性 patch/revert、
Apple 元数据、组态与大 dump、solver_accel/MG/predictor/telemetry 都不进入新提交。
原 RNG 分支和原工作区实物不删除；新分支从父节点重建，并非在污染提交之后删除文件。
因此新分支历史不含污染提交，但保留原分支仍会保留原仓库历史占用。

## 验证状态

CPU 单元测试与静态语法检查在本次整理时重新执行，详细覆盖见上述测试。
整理时完整 CPU 测试共 72 项通过；`git diff --check` 通过。
原始外部 Chroma 字段 dump 未在本次已检查的源快照中找到，未以自行生成数据冒充它。
本次未编译 Chroma dump 工具，未重新完成真实 MPI、GPU `momAction` 或两条 HMC 轨迹回归。
本机没有 CUDA/匹配的 QUDA 运行环境；未为测试替换 p100 当前安装或工作区。
发布的是可审查、可继续验证的干净功能分支，不是新的完整 A/B 等价性通过报告。
