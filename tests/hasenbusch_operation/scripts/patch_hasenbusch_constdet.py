from pathlib import Path


ACTION_FILES = [
    Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/pyquda/action/clover_hasenbusch.py"),
    Path("/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/pyquda_core/pyquda/action/clover_hasenbusch.py"),
]
RUNNER = Path(
    "/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32.py"
)


def replace_once(text: str, old: str, new: str, label: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"{label} not found in {path}")
    return text.replace(old, new, 1)


def patch_action(path: Path) -> None:
    text = path.read_text()
    text = replace_once(
        text,
        "        determinant_ratio: bool = False,\n        mu_prec: Optional[float] = None,\n",
        "        determinant_ratio: bool = False,\n        constdet: bool = False,\n        mu_prec: Optional[float] = None,\n",
        "init signature",
        path,
    )
    text = replace_once(
        text,
        "        self.mu_prec = mu_prec\n        self.determinant_ratio = determinant_ratio\n",
        (
            "        self.mu_prec = mu_prec\n"
            "        self.determinant_ratio = determinant_ratio\n"
            "        # Chroma 的 multihasen cancel 使用 const-det 预条件化：不包含 clover trlog/sigma-trace。\n"
            "        self.constdet = constdet\n"
        ),
        "constdet attr",
        path,
    )
    text = replace_once(
        text,
        (
            '        if self.determinant_ratio and self.mu_prec is None:\n'
            '            getLogger().critical("determinant_ratio=True requires mu_prec", ValueError)\n'
        ),
        (
            '        if self.determinant_ratio and self.mu_prec is None:\n'
            '            getLogger().critical("determinant_ratio=True requires mu_prec", ValueError)\n'
            '        if self.determinant_ratio and self.constdet:\n'
            '            getLogger().critical("determinant_ratio and constdet cannot both be True", ValueError)\n'
        ),
        "constdet validation",
        path,
    )
    text = replace_once(
        text,
        (
            "        else:\n"
            "            # detratio=False 的 QUDA force 会包含 clover determinant 的 sigma-trace 项；\n"
            "            # action 也必须包含同一个 trlogA，否则 finite-difference 与 force 不闭合。\n"
            "            self.invert_param.compute_clover_trlog = 1\n"
            "            self.updateClover()\n"
            "            self.invert_param.compute_clover_trlog = 0\n"
            "            self._solve(self.phi, self.mu, compute_action=True)\n"
            "            return (\n"
            "                self.invert_param.action[0]\n"
            "                - self.latt_info.volume // 2 * self.latt_info.Ns * self.latt_info.Nc\n"
            "                - self.multiplicity * self.invert_param.trlogA[1]\n"
            "            )\n"
        ),
        (
            "        else:\n"
            "            if not self.constdet:\n"
            "                # 普通 twisted-clover determinant force 包含 clover sigma-trace；action 也要包含 trlogA。\n"
            "                self.invert_param.compute_clover_trlog = 1\n"
            "                self.updateClover()\n"
            "                self.invert_param.compute_clover_trlog = 0\n"
            "            else:\n"
            "                self.updateClover()\n"
            "            self._solve(self.phi, self.mu, compute_action=True)\n"
            "            action = self.invert_param.action[0] - self.latt_info.volume // 2 * self.latt_info.Ns * self.latt_info.Nc\n"
            "            if not self.constdet:\n"
            "                action -= self.multiplicity * self.invert_param.trlogA[1]\n"
            "            return action\n"
        ),
        "action constdet branch",
        path,
    )
    text = replace_once(
        text,
        (
            "        else:\n"
            "            self._solve(self.phi, self.mu, compute_action=False)\n"
            "            self.source[0] = self.phi\n"
        ),
        (
            "        else:\n"
            "            self._solve(self.phi, self.mu, compute_action=False)\n"
            "            if self.constdet:\n"
            "                # const-det cancel 只需要伪费米子力；传入零 x0 让 QUDA 跳过 sigma-trace 且不添加 detratio source。\n"
            "                self.source.data = 0\n"
            "            else:\n"
            "                self.source[0] = self.phi\n"
        ),
        "force constdet source",
        path,
    )
    text = replace_once(
        text,
        "                int(self.determinant_ratio),\n            )\n",
        "                int(self.determinant_ratio or self.constdet),\n            )\n",
        "force detratio flag",
        path,
    )
    path.write_text(text)


def patch_runner(path: Path) -> None:
    text = path.read_text()
    text = replace_once(
        text,
        '    logger.info("light_has_cancel = determinant action at largest Hasenbusch mu")\n',
        '    logger.info("light_has_cancel = Chroma-style const-det cancel action at largest Hasenbusch mu")\n',
        "runner cancel log",
        path,
    )
    text = replace_once(
        text,
        (
            "        determinant_ratio=False,\n"
            "        force_sign=config[\"hasenbusch_force_sign\"],\n"
        ),
        (
            "        determinant_ratio=False,\n"
            "        constdet=True,\n"
            "        force_sign=config[\"hasenbusch_force_sign\"],\n"
        ),
        "runner has_cancel constdet",
        path,
    )
    path.write_text(text)


for action_file in ACTION_FILES:
    patch_action(action_file)
patch_runner(RUNNER)
print("patched_constdet_cancel")
