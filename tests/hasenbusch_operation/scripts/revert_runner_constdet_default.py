from pathlib import Path


RUNNER = Path(
    "/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/hasenbusch_operation/scripts/run_hmc_hasenbusch_12x32.py"
)

text = RUNNER.read_text()
text = text.replace(
    '    logger.info("light_has_cancel = Chroma-style const-det cancel action at largest Hasenbusch mu")\n',
    '    logger.info("light_has_cancel = determinant action at largest Hasenbusch mu")\n',
)
text = text.replace(
    '        determinant_ratio=False,\n        constdet=True,\n        force_sign=config["hasenbusch_force_sign"],\n',
    '        determinant_ratio=False,\n        force_sign=config["hasenbusch_force_sign"],\n',
)
RUNNER.write_text(text)
print("runner_constdet_default_reverted")
