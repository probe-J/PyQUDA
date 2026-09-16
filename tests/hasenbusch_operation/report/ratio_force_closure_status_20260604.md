# Hasenbusch ratio force 闭合进展 2026-06-04

## 当前结论

- 不应启动长 HMC；ratio_only force finite-difference 仍未闭合。
- Chroma 四项 ratio 结构已保留：`+M_prec(phi,X) +M_prec^dag(X,phi) -M^dag(X,Y) -M(Y,X)`。
- 四项结构将 pair4 从灾难性偏差降到同量级，但 full ratio chain 多 seed 仍不稳定。

## 本轮新增/修改

- 修改 `pyquda_core/pyquda/action/clover_hasenbusch.py`：
  - 保留四项 ratio force 实现。
  - 新增 `sym_odd_clover_scale` 对 `pc_quda_norm` 分支 active odd `A_oo^-1 insertion` 的诊断缩放；默认 `1.0`，不改变正式行为。
- 修改 `scripts/monomial_fd_breakdown_12x32.py`：
  - 去掉不存在的 `core.end()`，避免 CSV 已写出后报错。

## 已验证

- `python3 -m py_compile pyquda_core/pyquda/action/clover_hasenbusch.py` 通过。
- 逐 monomial action finite-difference 已生成：
  - `csv/monomial_fd_monomial_fd_ratio_seed10086_20260604g.csv`
  - `csv/monomial_fd_monomial_fd_ratio_seed10087_20260604f.csv`
- seed10087 force term 明细已生成：
  - `csv/force_terms_forcefd_ratio_fullchain_pc_quda_norm_fourterm_quarter_terms_seed10087_20260604g.csv`

## 关键数据

默认四项 `pc_quda_norm`：

- seed10086：逐 monomial FD 总和约 `1.28043`；默认 quarter force 约 `1.37243`，接近但未闭合。
- seed10087：逐 monomial FD 总和约 `-0.49719`；默认 quarter force 约 `1.09449`，符号/结构不闭合。

active_mu 尝试：

- 将 shift 项从 `self.mu` 改为 `invert_param.mu` 后三 seed 均变差，已回退。

`A_oo` 诊断缩放：

- `sym_odd_clover_scale=0` 未解决闭合：
  - seed10086：`fd=1.28109, force=0.70422`
  - seed10087：`fd=-0.49683, force=0.38843`
  - seed10088：`fd=0.74390, force=0.15706`

## 下一步建议

1. 不再调全局 force_output_scale。
2. 聚焦 `_add_sym_oddodd_quda_norm_deriv_force()` 内部 `A_oo^-1 insertion` 的左右矢量和 dagger 分支：
   - `PLUS A_oo^-1 insertion`
   - `MINUS A_oo^-1 insertion`
3. 用单项 force_term_filter 分别对这两个 insertion 做 operator-level probe，确认 QUDA pair-force 半格 odd parity 映射是否等价于 Chroma `derivUnprecOddOddLinOp`。
4. 闭合前不要跑 `t=1, stop=1000` 长 HMC。
