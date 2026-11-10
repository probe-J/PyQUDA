"""CPU 单测：配置/归一化/CLI 延迟导入及 MPI 输出目录竞态，不加载 GPU 库。"""

import copy
import importlib.util
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rng_run_hmc", HERE / "run_hmc.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = runner.load_config(HERE / "config_12x32.json")

    def test_baseline(self):
        c = self.config
        self.assertEqual((c["tol"], c["maxiter"], c["seed"]), (1e-10, 1000, 20260910))
        self.assertEqual((c["nsteps_outer"], c["nsteps_inner"]), (14, 2))
        self.assertAlmostEqual(c["t"], 0.1 / math.sqrt(2))
        self.assertEqual(c["strange_matpc_type"], "odd_odd")
        self.assertTrue(c["strange_action_rational"])

    def test_reject_bad_values(self):
        cases = {"tol": [0, float("nan"), float("inf"), True, "1e-10"],
                 "maxiter": [0, -1, 1.5], "latt_size": [[12, 12, 12], [12, 12, 12, 31]],
                 "seed": [-1, 2**47], "t": [0.1], "mu_scale": [1.0],
                 "clover_csw": [1.0], "nsteps_inner": [0], "strange_matpc_type": ["even_even"],
                 "hasenbusch_mu": [[0, .001, .001, .0035, .0075, .0155]]}
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    c = copy.deepcopy(self.config)
                    c[key] = value
                    with self.assertRaises(ValueError):
                        runner.validate_config(c)

    def test_missing_unknown_keys_and_rational(self):
        for key in ("t", "rational", "mu_scale"):
            c = copy.deepcopy(self.config)
            del c[key]
            with self.assertRaises(ValueError):
                runner.validate_config(c)
        c = copy.deepcopy(self.config)
        c["typo"] = 1
        with self.assertRaises(ValueError):
            runner.validate_config(c)
        c = copy.deepcopy(self.config)
        c["rational"]["offset_force"][0] = -1
        with self.assertRaises(ValueError):
            runner.validate_config(c)

    def test_template_coefficients_and_time(self):
        # 模板由另一 agent 维护；逐项比对数值，不以文件名推断归一化。
        root = ET.parse(HERE / "chroma_12x32.xml").getroot()
        c = self.config
        self.assertEqual(float(root.findtext(".//tau0")), c["chroma_tau0"])
        self.assertEqual([int(e.text) for e in root.findall(".//n_steps")], [14, 2])
        clovers = [float(e.text) for e in root.findall(".//clovCoeff")]
        self.assertEqual(clovers, [c["clover_csw"]] * 3 + [c["clover_csw_strange"]])
        for mode, path in (("force", ".//ForceApprox//PFECoeffs"),
                           ("sample", ".//ActionApprox//IPFECoeffs"),
                           ("action", ".//ActionApprox//PFECoeffs")):
            element = root.find(path)
            self.assertEqual(c["rational"]["norm_" + mode], float(element.findtext("norm")))
            for key, xml in (("residue", "res"), ("offset", "pole")):
                self.assertEqual(c["rational"][key + "_" + mode], list(map(float, element.findtext(xml).split())))

    def test_help_without_site_packages(self):
        result = subprocess.run([sys.executable, "-B", "-S", str(HERE / "run_hmc.py"), "--help"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--config", "--gauge", "--output-dir", "--arm", "--ntraj", "--use-chroma-accept", "--mom-action-only"):
            self.assertIn(flag, result.stdout)

    def test_cli_defaults_and_guard(self):
        argv = ["--gauge", "input.lime", "--output-dir", "output", "--arm", "native"]
        args = runner.parse_args(argv)
        self.assertEqual(args.ntraj, 2)
        self.assertFalse(args.use_chroma_accept)
        for extra in (["--use-chroma-accept"], ["--ntraj", "0"]):
            result = subprocess.run([sys.executable, "-B", "-S", str(HERE / "run_hmc.py"), *argv, *extra],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_config_load_without_gpu_imports(self):
        code = ("import runpy,sys; m=runpy.run_path(sys.argv[1]); m['load_config'](sys.argv[2]); "
                "assert not any(n.startswith(('pyquda','cupy','mpi4py')) for n in sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-S", "-c", code,
                                 str(HERE / "run_hmc.py"), str(HERE / "config_12x32.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mpi_output_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            root = Mock(rank=0)
            root.bcast.side_effect = lambda error, root: error
            runner.prepare_output(out, root)
            self.assertTrue((out / "quda-cache").is_dir())
            root.Barrier.assert_called_once()
            # 晚到的非根 rank 不能再次检查根 rank 已创建的产物。
            late = Mock(rank=1)
            late.bcast.return_value = None
            runner.prepare_output(out, late)
            late.Barrier.assert_called_once()
            with self.assertRaises(ValueError):
                runner.prepare_output(out, root)
            late.bcast.return_value = "output-dir must be empty"
            with self.assertRaises(ValueError):
                runner.prepare_output(out, late)

    def test_seed_probe_does_not_consume_native_rng(self):
        hmc = SimpleNamespace(random=random.Random(20260910))
        before = hmc.random.getstate()
        first = runner.rng_state(hmc, None)
        self.assertEqual(before, hmc.random.getstate())
        self.assertEqual(first["native_next_seed"], hmc.random.randrange(2**32))
        self.assertNotEqual(first["native_state_sha256"], runner.rng_state(hmc, None)["native_state_sha256"])

    def test_momentum_constant_and_native_scale(self):
        import numpy as np
        # 每个 link 选上三角 P01=2，故 -Tr(P^2)/2=4；四个 link 的零点抵消为 0。
        data = np.zeros((4, 10))
        data[:, 0] = 2
        mom = SimpleNamespace(getHost=lambda: data)
        hmc = Mock()
        hmc.momAction.return_value = 0.0
        comm = Mock()
        comm.allreduce.side_effect = lambda value: value
        result = runner.momentum_check(hmc, mom, comm, 1)
        self.assertEqual(result["host_minus_trace_p2_over_2"], 16)
        self.assertEqual(result["quda_constant"], -16)
        self.assertTrue(result["passed"])
        hmc.momAction.return_value = 16.0
        self.assertFalse(runner.momentum_check(hmc, mom, comm, 1)["passed"])


if __name__ == "__main__":
    unittest.main()
