from datetime import datetime
from pathlib import Path

ROOT = Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


def patch_action(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    MatQuda,\n    computeCloverPairForceV2Quda,\n",
        "    MatQuda,\n    cloverQuda,\n    computeCloverPairForceV2Quda,\n    dslashQuda,\n",
        "quda imports",
    )
    text = replace_once(
        text,
        "    QudaMatPCType,\n    QudaSolutionType,\n",
        "    QudaMatPCType,\n    QudaParity,\n    QudaSolutionType,\n",
        "enum imports",
    )
    text = replace_once(
        text,
        '        if self.ratio_force_mode not in ("quda_detratio", "pair"):\n'
        "            getLogger().critical(\"ratio_force_mode must be 'quda_detratio' or 'pair'\", ValueError)\n",
        '        if self.ratio_force_mode not in ("quda_detratio", "pair", "pc_pair"):\n'
        "            getLogger().critical(\"ratio_force_mode must be 'quda_detratio', 'pair' or 'pc_pair'\", ValueError)\n",
        "ratio mode validation",
    )
    text = replace_once(
        text,
        "        self.tmp = LatticeFermion(latt_info)\n"
        "        self.y = LatticeFermion(latt_info)\n"
        "        self.force_tmp = LatticeGauge(latt_info)\n",
        "        self.tmp = LatticeFermion(latt_info)\n"
        "        self.y = LatticeFermion(latt_info)\n"
        "        # pc_pair 模式使用这些临时场显式构造 Chroma e/o 预条件化导数四项。\n"
        "        self.pc_tmp1 = LatticeFermion(latt_info)\n"
        "        self.pc_tmp2 = LatticeFermion(latt_info)\n"
        "        self.pc_tmp3 = LatticeFermion(latt_info)\n"
        "        self.force_tmp = LatticeGauge(latt_info)\n",
        "temp fields",
    )

    helper = '''    def _parity_field(self, field: LatticeFermion, parity: QudaParity) -> LatticeFermion:
        # 返回只含一个 checkerboard 的全场；pair force 用它隔离 Chroma 的 PC block。
        out = LatticeFermion(self.latt_info)
        out.data = 0
        if parity == QudaParity.QUDA_EVEN_PARITY:
            out.even = field.even
        elif parity == QudaParity.QUDA_ODD_PARITY:
            out.odd = field.odd
        else:
            getLogger().critical("_parity_field only accepts even/odd parity", ValueError)
        return out

    def _dslash_block(
        self,
        out: LatticeFermion,
        inp: LatticeFermion,
        out_parity: QudaParity,
        dagger: QudaDagType,
        scale: float = 1.0,
    ) -> None:
        # Chroma 的 evenOdd/oddEven block 是 -0.5 * D.apply；scale 显式承载这个系数。
        old_dagger = self.invert_param.dagger
        self.invert_param.dagger = dagger
        out.data = 0
        if out_parity == QudaParity.QUDA_EVEN_PARITY:
            dslashQuda(out.even_ptr, inp.odd_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY)
            out.even = out.even * scale
        elif out_parity == QudaParity.QUDA_ODD_PARITY:
            dslashQuda(out.odd_ptr, inp.even_ptr, self.invert_param, QudaParity.QUDA_ODD_PARITY)
            out.odd = out.odd * scale
        else:
            getLogger().critical("_dslash_block only accepts even/odd parity", ValueError)
        self.invert_param.dagger = old_dagger

    def _clover_inv_even(self, out: LatticeFermion, inp: LatticeFermion, dagger: QudaDagType) -> None:
        # Chroma evenEvenInvLinOp 只作用在 even checkerboard。
        old_dagger = self.invert_param.dagger
        self.invert_param.dagger = dagger
        out.data = 0
        cloverQuda(out.even_ptr, inp.even_ptr, self.invert_param, QudaParity.QUDA_EVEN_PARITY, 1)
        self.invert_param.dagger = old_dagger

    def _add_eoprec_deriv_force(
        self,
        force: LatticeGauge,
        chi: LatticeFermion,
        psi: LatticeFermion,
        dagger: QudaDagType,
        sign: float,
        label: str,
    ) -> None:
        # 复刻 Chroma EvenOddPrecLogDetLinearOperator::deriv 的四个 block。
        msign = QudaDagType.QUDA_DAG_NO if dagger == QudaDagType.QUDA_DAG_YES else QudaDagType.QUDA_DAG_YES
        chi_odd = self._parity_field(chi, QudaParity.QUDA_ODD_PARITY)
        psi_odd = self._parity_field(psi, QudaParity.QUDA_ODD_PARITY)
        self._add_pair_force_v2(force, chi_odd, psi_odd, sign, f"{label}: A_oo prime")
        self._dslash_block(self.pc_tmp1, psi_odd, QudaParity.QUDA_EVEN_PARITY, dagger, -0.5)
        self._clover_inv_even(self.pc_tmp2, self.pc_tmp1, dagger)
        self._add_pair_force_v2(
            force,
            chi_odd,
            self._parity_field(self.pc_tmp2, QudaParity.QUDA_EVEN_PARITY),
            -sign * 0.5,
            f"{label}: -D_oe prime Aee^-1 D_eo",
        )
        self._dslash_block(self.pc_tmp1, chi_odd, QudaParity.QUDA_EVEN_PARITY, msign, -0.5)
        self._clover_inv_even(self.pc_tmp3, self.pc_tmp1, msign)
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
            self._parity_field(self.pc_tmp2, QudaParity.QUDA_EVEN_PARITY),
            sign,
            f"{label}: +A_ee prime insertion",
        )
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp3, QudaParity.QUDA_EVEN_PARITY),
            psi_odd,
            -sign * 0.5,
            f"{label}: -D_eo prime",
        )

'''
    text = replace_once(text, "    def _add_pair_force_v2(\n", helper + "    def _add_pair_force_v2(\n", "pc helpers")

    old_branch = '''        if self.determinant_ratio and force_out is not None and self.ratio_force_mode == "pair":
            # pair 模式保留为调试入口；默认不使用，因为它需要完整的 PC bilinear 重构。
            self._add_pair_force_v2(force_out, self.phi, self.quark[0], +1.0, "D_prec(phi,X)")
            self._add_pair_force_v2(force_out, self.quark[0], self.y, -1.0, "D(X,Y)")
            self._add_pair_force_v2(force_out, self.y, self.quark[0], -1.0, "D(Y,X)")
            self._add_pair_force_v2(force_out, self.quark[0], self.phi, +1.0, "D_prec(X,phi)")
        else:
'''
    new_branch = '''        if self.determinant_ratio and force_out is not None and self.ratio_force_mode == "pair":
            # pair 模式保留为调试入口；默认不使用，因为它没有 PC block 重构。
            self._add_pair_force_v2(force_out, self.phi, self.quark[0], +1.0, "D_prec(phi,X)")
            self._add_pair_force_v2(force_out, self.quark[0], self.y, -1.0, "D(X,Y)")
            self._add_pair_force_v2(force_out, self.y, self.quark[0], -1.0, "D(Y,X)")
            self._add_pair_force_v2(force_out, self.quark[0], self.phi, +1.0, "D_prec(X,phi)")
        elif self.determinant_ratio and force_out is not None and self.ratio_force_mode == "pc_pair":
            # pc_pair 显式对应 Chroma ratio force：D_prec(phi,X)-D(X,Y)-D(Y,X)+D_prec(X,phi)。
            old_mu_for_pc = self.invert_param.mu
            self.invert_param.mu = self.mu_prec
            self._add_eoprec_deriv_force(force_out, self.phi, self.quark[0], QudaDagType.QUDA_DAG_NO, +1.0, "D_prec(phi,X)")
            self._add_eoprec_deriv_force(force_out, self.quark[0], self.phi, QudaDagType.QUDA_DAG_YES, +1.0, "D_prec(X,phi)")
            self.invert_param.mu = self.mu
            self._add_eoprec_deriv_force(force_out, self.quark[0], self.y, QudaDagType.QUDA_DAG_YES, -1.0, "-D(X,Y)")
            self._add_eoprec_deriv_force(force_out, self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, -1.0, "-D(Y,X)")
            self.invert_param.mu = old_mu_for_pc
        else:
'''
    text = replace_once(text, old_branch, new_branch, "pc_pair branch")
    path.write_text(text, encoding="utf-8")


def patch_runner(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace('choices=["quda_detratio", "pair"],', 'choices=["quda_detratio", "pair", "pc_pair"],')
    text = text.replace(
        "help=\"ratio force 实现：默认使用 QUDA detratio；pair 仅用于调试 PC bilinear force。\",",
        "help=\"ratio force 实现：默认 QUDA detratio；pair/pc_pair 用于调试 bilinear 和 PC block force。\",",
    )
    text = text.replace(
        "help=\"ratio force 实现：默认 QUDA detratio；pair 用于调试新增 bilinear 接口。\",",
        "help=\"ratio force 实现：默认 QUDA detratio；pair/pc_pair 用于调试 bilinear 和 PC block force。\",",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    files = [
        ROOT / "pyquda/action/clover_hasenbusch.py",
        ROOT / "pyquda_core/pyquda/action/clover_hasenbusch.py",
        ROOT / "tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32.py",
        ROOT / "tests/hasenbusch_operation/scripts/force_check_hasenbusch_12x32.py",
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in files:
        backup = path.with_name(path.name + f".codex_bak_pc_pair_{stamp}")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    for path in files[:2]:
        patch_action(path)
    for path in files[2:]:
        patch_runner(path)
    print(f"patched pc_pair, backup stamp={stamp}")


if __name__ == "__main__":
    main()
