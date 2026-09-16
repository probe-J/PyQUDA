from datetime import datetime
from pathlib import Path


ROOT = Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA")
FILES = [
    ROOT / "pyquda/action/clover_hasenbusch.py",
    ROOT / "pyquda_core/pyquda/action/clover_hasenbusch.py",
]

NEW_BLOCK = '''    def _active_parity(self) -> QudaParity:
        # PyQUDA 当前使用 even-even asymmetric；保留 odd-odd 分支便于后续对照 Chroma 文档公式。
        if self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC:
            return QudaParity.QUDA_EVEN_PARITY
        if self.invert_param.matpc_type == QudaMatPCType.QUDA_MATPC_ODD_ODD_ASYMMETRIC:
            return QudaParity.QUDA_ODD_PARITY
        getLogger().critical("pc_pair only supports asymmetric even-even or odd-odd matpc", NotImplementedError)

    def _other_parity(self, parity: QudaParity) -> QudaParity:
        if parity == QudaParity.QUDA_EVEN_PARITY:
            return QudaParity.QUDA_ODD_PARITY
        if parity == QudaParity.QUDA_ODD_PARITY:
            return QudaParity.QUDA_EVEN_PARITY
        getLogger().critical("_other_parity only accepts even/odd parity", ValueError)

    def _clover_inv_parity(
        self, out: LatticeFermion, inp: LatticeFermion, parity: QudaParity, dagger: QudaDagType
    ) -> None:
        # Chroma 的 A^{-1} block 是未 shifted 的 base clover inverse，作用在 inactive checkerboard。
        old_dagger = self.invert_param.dagger
        old_dslash_type = self.invert_param.dslash_type
        self.invert_param.dagger = dagger
        self.invert_param.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
        out.data = 0
        if parity == QudaParity.QUDA_EVEN_PARITY:
            cloverQuda(out.even_ptr, inp.even_ptr, self.invert_param, parity, 1)
        elif parity == QudaParity.QUDA_ODD_PARITY:
            cloverQuda(out.odd_ptr, inp.odd_ptr, self.invert_param, parity, 1)
        else:
            getLogger().critical("_clover_inv_parity only accepts even/odd parity", ValueError)
        self.invert_param.dslash_type = old_dslash_type
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
        # 复刻 e/o PC 导数：A'_aa - D'_ai Aii^-1 D_ia + Aii' 插入项 - D'_ia。
        active = self._active_parity()
        inactive = self._other_parity(active)
        msign = QudaDagType.QUDA_DAG_NO if dagger == QudaDagType.QUDA_DAG_YES else QudaDagType.QUDA_DAG_YES
        chi_active = self._parity_field(chi, active)
        psi_active = self._parity_field(psi, active)

        self._add_pair_force_v2(force, chi_active, psi_active, sign, f"{label}: A_active prime")
        self._dslash_block(self.pc_tmp1, psi_active, inactive, dagger, -0.5)
        self._clover_inv_parity(self.pc_tmp2, self.pc_tmp1, inactive, dagger)
        self._add_pair_force_v2(
            force,
            chi_active,
            self._parity_field(self.pc_tmp2, inactive),
            -sign * 0.5,
            f"{label}: -D_active,inactive prime Aii^-1 D_inactive,active",
        )

        self._dslash_block(self.pc_tmp1, chi_active, inactive, msign, -0.5)
        self._clover_inv_parity(self.pc_tmp3, self.pc_tmp1, inactive, msign)
        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp3, inactive),
            self._parity_field(self.pc_tmp2, inactive),
            sign,
            f"{label}: +A_inactive prime insertion",
        )

        self._add_pair_force_v2(
            force,
            self._parity_field(self.pc_tmp3, inactive),
            psi_active,
            -sign * 0.5,
            f"{label}: -D_inactive,active prime",
        )

'''


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        path.with_name(path.name + f".codex_bak_active_parity_{stamp}").write_text(text, encoding="utf-8")
        start = text.index("    def _clover_inv_even(")
        end = text.index("    def _add_pair_force_v2(", start)
        path.write_text(text[:start] + NEW_BLOCK + text[end:], encoding="utf-8")
    print(f"patched active parity pc_pair, backup stamp={stamp}")


if __name__ == "__main__":
    main()
