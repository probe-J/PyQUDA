"""生产头 HMC 的 CPU 回归：配置、factory、按列读场和 MG 证据门禁。

在仓库根目录运行 python -B -m unittest discover -s tests/rng -p test_header_hmc.py -v。
输入为本文件构造的小字典、NumPy 数组和合成 telemetry；只输出 unittest 结果。
AST 加载真实 factory，替换 action/HMC/QUDA/MPI，不依赖外部成功配置或 GPU。
uniform 仅检查 factory 参数传递，不重复 action 内部多极点残差测试。
"""

import ast
from contextlib import ExitStack, contextmanager, redirect_stderr
import copy
from dataclasses import dataclass
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
FACTORY_PATH = HERE.parents[1] / "pyquda_core/pyquda/hasenbusch.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPORT = load_file("header_test_support", HERE / "chroma_hmc_support.py")
with patch.dict(sys.modules, {"chroma_hmc_support": SUPPORT}):
    RUNNER = load_file("header_test_runner", HERE / "run_header_hmc.py")


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def header_config():
    """独立构造两种 logdet、一个 ratio 副本及 cancel/strange，不读取 campaign。"""
    rational = {}
    for mode in ("sample", "action", "force"):
        rational.update({f"norm_{mode}": 0.5, f"residue_{mode}": [0.25, 0.75],
                         f"offset_{mode}": [0.01, 0.1]})
    ratio = dict(id="ratio_h", role="hasenbusch_ratio", mass=-0.3, mass_precond=-0.2,
                 tol=1e-12, maxiter=100)
    rows = [dict(id="gauge", role="gauge"),
            dict(id="light_log", role="logdet", mass=-0.3, n_flavor=2),
            dict(id="strange_log", role="logdet", mass=-0.05, n_flavor=1),
            ratio, dict(ratio, id="ratio_md", tol=1e-8),
            dict(id="cancel", role="hasenbusch_cancel", mass=-0.2, tol=1e-10, maxiter=200),
            dict(id="strange", role="strange", mass=-0.05, n_flavor=1, rational=rational)]
    seed = [1, 2, 3, 4]
    return dict(
        latt_size=[2, 2, 2, 4], t_boundary=-1, seed_words=seed,
        seed_packed=sum(n << (12 * i) for i, n in enumerate(seed)), beta=6.0, u0=0.9,
        clover_csw_light=1.1, clover_csw_strange=1.3,
        stout=dict(n_smear=1, rho=0.1, dir_ignore=-1), monomials=rows,
        copy_list=[dict(copyFrom="ratio_h", copyTo="ratio_md")],
        sampling_order=[f"momentum[{d}]" for d in range(4)] + ["strange", "cancel", "ratio_md"],
        integrator=dict(outer_type="LCM_STS_FORCE_GRAD", inner_type="LCM_STS_FORCE_GRAD",
                        outer=3, inner=2, tau0=1.0,
                        inner_monomials=["gauge", "strange_log", "light_log"],
                        outer_monomials=["ratio_md", "cancel", "strange"]),
    )


def set_nested(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


class NormalizeHeaderTests(unittest.TestCase):
    def test_explicit_alias_merges_tolerances_and_preserves_order_without_mutating_input(self):
        source = header_config()
        before = copy.deepcopy(source)
        c = SUPPORT.normalize_config(source)
        self.assertEqual(source, before)
        self.assertEqual([r["id"] for r in c["monomials"]],
                         ["gauge", "light_log", "strange_log", "ratio_h", "cancel", "strange"])
        self.assertEqual(c["aliases"], {r["id"]: "ratio_h" if r["id"] == "ratio_md" else r["id"]
                                        for r in source["monomials"]})
        rows = {r["id"]: r for r in c["monomials"]}
        self.assertEqual((rows["ratio_h"]["role"], rows["ratio_h"]["mass"],
                          rows["ratio_h"]["mass_precond"]), ("ratio", -0.3, -0.2))
        self.assertEqual((rows["ratio_h"]["action_tol"], rows["ratio_h"]["md_tol"]), (1e-12, 1e-8))
        self.assertEqual(rows["cancel"]["role"], "cancel")
        self.assertEqual([rows[k]["clover_csw"] for k in ("light_log", "strange_log", "strange")],
                         [1.1, 1.3, 1.3])
        self.assertEqual(c["integrator"]["inner_monomials"], ["gauge", "strange_log", "light_log"])
        self.assertEqual(c["integrator"]["outer_monomials"], ["ratio_h", "cancel", "strange"])
        self.assertEqual(c["sampling_order"], ["strange", "cancel", "ratio_h"])
        self.assertEqual(c["stout"]["dir_ignore"], 4)

    def test_no_copy_list_never_implicitly_merges_identical_physics(self):
        c = header_config()
        del c["copy_list"]
        c["sampling_order"].append("ratio_h")
        c["integrator"]["outer_monomials"].append("ratio_h")
        out = SUPPORT.normalize_config(c)
        self.assertEqual(len(out["monomials"]), 7)
        self.assertEqual(out["aliases"]["ratio_md"], "ratio_md")
        self.assertEqual(out["sampling_order"], ["strange", "cancel", "ratio_md", "ratio_h"])

    def test_duplicate_pseudofermion_sampling_is_rejected_after_alias_resolution(self):
        # 同名重复和显式 H/MD 副本都共享一次采样，不能静默改变原始刷新序列。
        for duplicate in ("strange", "cancel", "ratio_md", "ratio_h"):
            with self.subTest(duplicate=duplicate):
                c = header_config()
                c["sampling_order"].append(duplicate)
                with self.assertRaisesRegex(ValueError, "sampling_order"):
                    SUPPORT.normalize_config(c)

    def test_momentum_refresh_requires_all_four_directions_in_order_at_start(self):
        pseudofermions = ["strange", "cancel", "ratio_md"]
        cases = [["momentum[0]"], ["momentum[1]", "momentum[0]", "momentum[2]", "momentum[3]"],
                 [f"momentum[{d}]" for d in range(4)] + ["momentum[0]"]]
        for prefix in cases:
            with self.subTest(prefix=prefix):
                c = header_config()
                c["sampling_order"] = prefix + pseudofermions
                with self.assertRaisesRegex(ValueError, "momentum"):
                    SUPPORT.normalize_config(c)
        c = header_config()
        c["sampling_order"] = pseudofermions + [f"momentum[{d}]" for d in range(4)]
        with self.assertRaisesRegex(ValueError, "momentum"):
            SUPPORT.normalize_config(c)

    def test_repeated_layer_action_is_rejected_including_explicit_alias(self):
        for layer, duplicate in (("inner_monomials", "gauge"), ("outer_monomials", "ratio_md"),
                                 ("outer_monomials", "ratio_h")):
            with self.subTest(layer=layer, duplicate=duplicate):
                c = header_config()
                c["integrator"][layer].append(duplicate)
                with self.assertRaisesRegex(ValueError, "exactly one integrator level"):
                    SUPPORT.normalize_config(c)

    def test_missing_copy_list_requires_independent_sampling(self):
        c = header_config()
        del c["copy_list"]
        c["integrator"]["outer_monomials"].append("ratio_h")
        c["sampling_order"] = ["ratio_h", "strange", "cancel"]
        with self.assertRaisesRegex(ValueError, "sampling_order"):
            SUPPORT.normalize_config(c)

    def test_copy_physical_differences_are_rejected(self):
        for key, value in (("role", "cancel"), ("mass", -0.31), ("mass_precond", -0.19),
                           ("clover_csw", 1.2), ("n_flavor", 2), ("rational", {"norm_force": 1})):
            with self.subTest(key=key):
                c = header_config()
                c["monomials"][4][key] = value
                with self.assertRaisesRegex(ValueError, "inconsistent HMC/MD copies"):
                    SUPPORT.normalize_config(c)

    def test_ambiguous_copy_lists_and_duplicate_ids_are_rejected(self):
        pairs = [("ratio_h", "absent"), ("absent", "ratio_md"), ("ratio_h", "ratio_h"),
                 ("ratio_h", "ratio_md"), ("ratio_md", "cancel"), ("cancel", "ratio_md")]
        for source, dest in pairs:
            with self.subTest(source=source, dest=dest):
                c = header_config()
                c["copy_list"].append(dict(copyFrom=source, copyTo=dest))
                with self.assertRaisesRegex(ValueError, "copyList"):
                    SUPPORT.normalize_config(c)
        c = header_config()
        c["monomials"].append(copy.deepcopy(c["monomials"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate monomial"):
            SUPPORT.normalize_config(c)

    def test_invalid_layer_and_sampling_coverage_are_rejected(self):
        cases = [("integrator", "inner_monomials", ["gauge", "strange_log", "light_log", "ratio_h"]),
                 ("integrator", "outer_monomials", ["ratio_md", "strange"]),
                 ("sampling_order", None, ["strange", "ratio_md"]),
                 ("sampling_order", None, ["strange", "ratio_md", "cancel", "light_log"])]
        for key, subkey, value in cases:
            with self.subTest(key=key, subkey=subkey):
                c = header_config()
                set_nested(c, (key, subkey) if subkey else (key,), value)
                with self.assertRaises(ValueError):
                    SUPPORT.normalize_config(c)

    def test_solver_mass_and_copy_tolerance_errors_are_rejected(self):
        cases = [(3, "mass", -4), (3, "mass_precond", -0.3), (3, "tol", True),
                 (3, "tol", 0), (3, "maxiter", 1.5), (4, "maxiter", 101),
                 (3, "action_tol", [1e-12]), (4, "md_tol", 1), (5, "role", "unknown")]
        for index, key, value in cases:
            with self.subTest(index=index, key=key):
                c = header_config()
                c["monomials"][index][key] = value
                with self.assertRaises(ValueError):
                    SUPPORT.normalize_config(c)

    def test_seed_integrator_boundary_and_geometry_validation(self):
        cases = [(('seed_words',), [1, 2, 3]), (('seed_words',), [True, 2, 3, 4]),
                 (('seed_words',), [4096, 2, 3, 4]), (('seed_words',), [1, 2, 3, 2048]),
                 (('seed_words',), [-1, 2, 3, 4]), (('seed_packed',), 0),
                 (('integrator', 'outer_type'), 'LEAPFROG'), (('integrator', 'inner_type'), 'OTHER'),
                 (('integrator', 'outer'), 0), (('integrator', 'inner'), True),
                 (('integrator', 'outer'), 1.5), (('integrator', 'tau0'), 0),
                 (('t_boundary',), 0), (('t_boundary',), 2),
                 (('latt_size',), [2, 2, 2]), (('latt_size',), [2, 2, 2, 3]),
                 (('latt_size',), [2, 2, 2, -2]), (('latt_size',), [True, 2, 2, 4]),
                 (('beta',), 0), (('u0',), 1.1), (('stout', 'rho'), -1)]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                c = header_config()
                set_nested(c, path, value)
                with self.assertRaises(ValueError):
                    SUPPORT.normalize_config(c)
        c = header_config()
        c['t_boundary'] = 1
        self.assertEqual(SUPPORT.normalize_config(c)['t_boundary'], 1)

    def test_nonfinite_values_are_rejected_recursively(self):
        paths = [('beta',), ('t_boundary',), ('seed_packed',), ('integrator', 'tau0'),
                 ('stout', 'rho'), ('monomials', 3, 'mass'),
                 ('monomials', 6, 'rational', 'residue_force', 0)]
        for path in paths:
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(path=path, value=value):
                    c = header_config()
                    set_nested(c, path, value)
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        SUPPORT.normalize_config(c)


class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.constructors = {}

        def constructor(kind):
            def create(*args, **kwargs):
                result = SimpleNamespace(kind=kind, dirac=SimpleNamespace(setPrecision=Mock()))
                if kind == "ratio":
                    result.numer = SimpleNamespace(dirac=SimpleNamespace(setPrecision=Mock()))
                    result.denom = SimpleNamespace(dirac=SimpleNamespace(setPrecision=Mock()))
                if kind == "strange":
                    result._sym_force_helper = SimpleNamespace(dirac=SimpleNamespace(setPrecision=Mock()))
                return result
            self.constructors[kind] = Mock(side_effect=create)
            return self.constructors[kind]

        def hmc(li, actions, integrator, inner=None, **kwargs):
            return SimpleNamespace(li=li, actions=actions, integrator=integrator,
                                   hmc_inner=inner, stout=kwargs, _solverAccelerators=Mock(return_value=iter(())))

        self.hmc = Mock(side_effect=hmc)
        factory_module = module("pyquda.hasenbusch", __package__="pyquda", dataclass=dataclass,
                  GaugeAction=constructor("gauge"), CloverWilsonLogDetAction=constructor("logdet"),
                  CloverWilsonHasenbuschTwistAction=constructor("cancel"),
                  CloverWilsonAction=constructor("strange"), RationalParam=lambda **kw: SimpleNamespace(**kw),
                  HMC=self.hmc)
        # dataclass 解析延迟注解时会回查 sys.modules，AST 模块也必须先注册。
        patcher = patch.dict(sys.modules, {"pyquda.hasenbusch": factory_module})
        patcher.start()
        self.addCleanup(patcher.stop)
        ns = factory_module.__dict__
        ratio = constructor("ratio")
        tree = ast.parse(FACTORY_PATH.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ("build_clover_hmc_from_monomials", "StoutHMCConfig")]
        code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(FACTORY_PATH), "exec"), ns)
        self.factory = ns["build_clover_hmc_from_monomials"]
        self.stout_cls = ns["StoutHMCConfig"]
        self.steps = Mock(side_effect=lambda n: SimpleNamespace(steps=n))
        self.loop = Mock(return_value=object())
        modules = {
            "pyquda.action.clover_mass_ratio": module("pyquda.action.clover_mass_ratio",
                CloverWilsonScalarMassRatioAction=ratio),
            "pyquda.enum_quda": module("pyquda.enum_quda",
                QudaPrecision=SimpleNamespace(QUDA_DOUBLE_PRECISION="double")),
            "pyquda.hasenbusch": factory_module,
            "pyquda.hmc": module("pyquda.hmc", STSForceGrad=self.steps),
            "pyquda_utils.hmc_param": module("pyquda_utils.hmc_param", symanzikTreeGaugeLoopParam=self.loop),
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.config = SUPPORT.normalize_config(header_config())
        self.li = object()
        self.accel = dict(plain_precision="double", multigrid=dict(enabled=True, force_gradient=True))

    def build(self, **changes):
        c = self.config
        args = dict(latt_info=self.li, monomials=c["monomials"], gauge_loop_param=object(), beta=c["beta"],
                    inner_ids=c["integrator"]["inner_monomials"], outer_ids=c["integrator"]["outer_monomials"],
                    sampling_ids=c["sampling_order"], outer_integrator=object(), inner_integrator=object(),
                    stout=self.stout_cls(**c["stout"]), solver_accel=self.accel)
        args.update(changes)
        return self.factory(**args)

    def test_scalar_bare_masses_logdet_flavors_and_cancel_contract(self):
        _, actions = self.build()
        call = self.constructors["ratio"].call_args
        self.assertEqual(call.args, (self.li, -0.3, -0.2, 1e-12, 100, 1.1))
        self.assertEqual((call.kwargs["md_tol"], call.kwargs["action_tol"]), (1e-8, 1e-12))
        logs = self.constructors["logdet"].call_args_list
        self.assertEqual([c.args for c in logs], [(self.li, -0.3, 2, 1e-12, 10000, 1.1),
                                                 (self.li, -0.05, 1, 1e-12, 10000, 1.3)])
        cancel = self.constructors["cancel"].call_args
        self.assertEqual(cancel.args, (self.li, -0.2, 0.0, 1e-10, 200, 1.1))
        self.assertTrue(cancel.kwargs["constdet"])
        self.assertFalse(cancel.kwargs["determinant_ratio"])
        self.assertEqual(cancel.kwargs["matpc_type"], "odd_odd")
        self.assertEqual(cancel.kwargs["cancel_force_mode"], "pc_quda_norm")
        self.assertEqual(cancel.kwargs["cancel_sample_scale"], 1.0)
        self.assertEqual(actions["cancel"].quda_norm_force_scale, 1.0)

    def test_layer_sampling_order_precision_and_uniform_argument(self):
        outer, actions = self.build()
        self.assertEqual([a.label for a in outer.actions], ["ratio_h", "cancel", "strange"])
        self.assertEqual([a.label for a in outer.hmc_inner.actions], ["gauge", "strange_log", "light_log"])
        self.assertEqual(outer.chroma_sampling_order, [actions[k] for k in ("strange", "cancel", "ratio_h")])
        self.assertEqual(outer.stout, outer.hmc_inner.stout)
        self.assertEqual(outer.stout, dict(stout_n_smear=1, stout_rho=0.1,
                                           stout_dir_ignore=4, stout_smear_anisotropy=1.0))
        for name, action in actions.items():
            if name == "gauge":
                action.dirac.setPrecision.assert_not_called()
                continue
            children = [action] + [getattr(action, key) for key in ("numer", "denom", "_sym_force_helper")
                                   if hasattr(action, key)]
            for child in children:
                child.dirac.setPrecision.assert_called_once_with(
                    cuda="double", sloppy="double", refinement_sloppy="double",
                    precondition="double", eigensolver="double")
        strange = self.constructors["strange"].call_args
        self.assertEqual(strange.kwargs, dict(use_action_rational=True, matpc_type="odd_odd",
                                             multishift_tolerance="uniform"))
        self.assertEqual(vars(strange.args[1]), self.config["monomials"][-1]["rational"])

    def test_solver_config_is_forwarded_once_only_to_ratio_and_cancel(self):
        before = copy.deepcopy(self.accel)
        self.build()
        for kind in ("ratio", "cancel"):
            self.constructors[kind].assert_called_once()
            self.assertIs(self.constructors[kind].call_args.kwargs["solver_accel"], self.accel)
        for kind in ("gauge", "logdet", "strange"):
            for call in self.constructors[kind].call_args_list:
                self.assertNotIn("solver_accel", call.kwargs)
        self.assertEqual(self.accel, before)

    def test_duplicate_and_missing_declarations_fail_before_construction(self):
        rows = self.config["monomials"]
        cases = [dict(monomials=rows + [copy.deepcopy(rows[0])]),
                 dict(outer_ids=["ratio_h", "cancel", "strange", "ratio_h"]),
                 dict(inner_ids=["gauge", "strange_log", "light_log", "ratio_h"]),
                 dict(outer_ids=["ratio_h", "strange"]),
                 dict(outer_ids=["ratio_h", "cancel", "missing"]),
                 dict(sampling_ids=["strange", "cancel", "ratio_h", "ratio_h"]),
                 dict(sampling_ids=["strange", "ratio_h"]),
                 dict(sampling_ids=["strange", "cancel", "ratio_h", "light_log"]),
                 dict(monomials=[dict(r, role="invalid") if r["id"] == "gauge" else r for r in rows])]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build(**changes)
        self.hmc.assert_not_called()
        for ctor in self.constructors.values():
            ctor.assert_not_called()

    def test_runner_build_delegates_and_restores_alias_identity(self):
        outer, actions = RUNNER.build(self.li, self.config, self.accel)
        self.assertIs(actions["ratio_md"], actions["ratio_h"])
        self.assertEqual(len(actions), 7)
        self.loop.assert_called_once_with(0.9)
        self.assertEqual([c.args for c in self.steps.call_args_list], [(3,), (2,)])
        self.assertEqual(outer.integrator.steps, 3)
        self.assertEqual(outer.hmc_inner.integrator.steps, 2)
        outer._solverAccelerators.assert_called_once_with()
        self.assertIs(self.constructors["ratio"].call_args.kwargs["solver_accel"], self.accel)


class RunnerConfigTests(unittest.TestCase):
    def test_expected_md_tolerances_use_normalized_physical_action_labels(self):
        # 执行 runner 的真实映射，确保 H/MD 别名合并后的目标送到两次证据验收。
        tree = ast.parse((HERE / 'run_header_hmc.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        statement = next(n for n in main.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'expected_md_tols' for t in n.targets))
        ns = dict(c=SUPPORT.normalize_config(header_config()))
        exec(compile(ast.Module(body=[statement], type_ignores=[]), str(HERE / 'run_header_hmc.py'), 'exec'), ns)
        self.assertEqual(ns['expected_md_tols'], {'ratio_h:numer': 1e-8, 'cancel': 1e-10})
        checks = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'verify_mg_evidence']
        self.assertEqual(len(checks), 2)
        for check in checks:
            self.assertTrue(any(k.arg == 'expected_md_tols' and isinstance(k.value, ast.Name)
                                and k.value.id == 'expected_md_tols' for k in check.keywords))

    def test_gate_reset_destroys_shared_and_nested_bases_before_first_trajectory(self):
        # 复用 CPU QUDA 桩，执行真实 HMC/manager 生命周期和 runner 的 gate 后片段。
        from test_solver_accel import TestSolverAccel, load_methods
        case = TestSolverAccel()
        case.setUp()
        self.addCleanup(case.doCleanups)
        s = case.s
        numer = case.manager
        denom = s.SolverAcceleration(SimpleNamespace(latt_info=case.li, label='ratio:denom',
                                                     dirac=case.action.dirac), numer.config)
        cancel = s.SolverAcceleration(SimpleNamespace(latt_info=case.li, label='cancel',
                                                      dirac=case.action.dirac), numer.config)
        scalar_path = FACTORY_PATH.parent / 'action/clover_mass_ratio.py'
        scalar_tree = ast.parse(scalar_path.read_text())
        aggregate = next(n for n in scalar_tree.body if isinstance(n, ast.ClassDef)
                         and n.name == '_ScalarSolverManagers')
        ns = dict(ExitStack=ExitStack, contextmanager=contextmanager)
        exec(compile(ast.Module(body=[aggregate], type_ignores=[]), str(scalar_path), 'exec'), ns)
        hmc_cls = load_methods(FACTORY_PATH.parent / 'hmc.py', 'HMC',
                               {'_solverAccelerators', 'closeSolverAccel', 'configureSolverAccel'},
                               {'__name__': s.__package__ + '.hmc', '__package__': s.__package__})
        root = hmc_cls()
        root.stout_config = None
        root.fermion_monomials = [SimpleNamespace(solver_accel=ns['_ScalarSolverManagers'](numer, denom))]
        root.hmc_inner = SimpleNamespace(fermion_monomials=[SimpleNamespace(solver_accel=cancel)],
                                        stout_config=None, hmc_inner=None)
        root.loadGauge, root.initializeRNG = Mock(), Mock()
        root.closeSolverAccel = Mock(wraps=root.closeSolverAccel)
        root.configureSolverAccel = Mock(wraps=root.configureSolverAccel)
        config = s.asdict(numer.config)
        tree = ast.parse((HERE / 'run_header_hmc.py').read_text())
        gate = next(n for n in ast.walk(tree) if isinstance(n, ast.Try)
                    and any(isinstance(v, ast.Expr) and isinstance(v.value, ast.Call)
                            and isinstance(v.value.func, ast.Name) and v.value.func.id == 'force_gate'
                            for v in n.body))
        end = next(i for i, n in enumerate(gate.body) if isinstance(n, ast.For))
        start = next(i for i, n in enumerate(gate.body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'handle is not None')
        reset = compile(ast.Module(body=gate.body[start:end], type_ignores=[]),
                        str(HERE / 'run_header_hmc.py'), 'exec')
        param_denom = copy.copy(case.param)
        param_denom.mass += 0.1
        for arm in ('native', 'chroma'):
            with self.subTest(arm=arm):
                numer.solve(case.out, case.rhs, case.param, case.performance)
                cancel.solve(case.out, case.rhs, case.param, case.performance)
                denom.solve(case.out, case.rhs, param_denom, case.performance)
                self.assertIs(numer._entry, cancel._entry)
                bases = [numer._mg, denom._mg]
                for manager in (numer, denom, cancel):
                    self.assertTrue(manager._used)
                    manager._refresh_pending = True
                handle = Mock() if arm == 'chroma' else None
                attach = Mock(return_value=Mock())
                calls = Mock()
                for name in ('closeSolverAccel', 'configureSolverAccel', 'loadGauge', 'initializeRNG'):
                    calls.attach_mock(getattr(root, name), name)
                if handle is not None:
                    calls.attach_mock(handle.detach, 'detach')
                calls.attach_mock(attach, 'attach')
                gauge = object()
                scope = dict(hmc=root, accel=config, gauge=gauge, c={'seed_packed': 123}, handle=handle,
                             args=SimpleNamespace(arm=arm), attach_chroma_rng=attach)
                exec(reset, scope)
                names = [c[0] for c in calls.mock_calls]
                self.assertEqual(names, (['detach'] if handle is not None else []) +
                                 ['closeSolverAccel', 'configureSolverAccel', 'loadGauge', 'initializeRNG'] +
                                 (['attach'] if handle is not None else []))
                self.assertFalse(s._MG_POOL)
                for basis in bases:
                    basis.destroy.assert_called_once_with()
                for manager in (numer, denom, cancel):
                    self.assertIsNone(manager._entry)
                    self.assertFalse(manager._refresh_pending)
                    self.assertIsNone(manager.last_solve)
                    self.assertIsNone(manager.last_validation)
                    self.assertEqual(s.asdict(manager.config), config)
                    # 保留首次求解后的精度保护；原配置重新 configure 仍须成功。
                    self.assertTrue(manager._used)
                root.loadGauge.assert_called_with(gauge)
                root.initializeRNG.assert_called_with(123)
                if handle is not None:
                    attach.assert_called_once_with(root, 123, precision='double', use_chroma_accept=True)
                else:
                    attach.assert_not_called()
                # close/reconfigure 不预热；首次正式 MD 创建新基，不消费 gate 的 refresh。
                numer.solve(case.out, case.rhs, case.param, case.performance)
                self.assertIsNot(numer._mg, bases[0])
                self.assertEqual(numer._mg.updates, [])
                root.closeSolverAccel()

    def test_grid_validation_rejects_bad_length_type_partition_and_rank_count(self):
        # 直接执行 main 的网格检查片段；不运行输出目录准备或 GPU 初始化。
        tree = ast.parse((HERE / 'run_header_hmc.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        index = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'grid' for t in n.targets))
        self.assertIsInstance(main.body[index + 1], ast.If)
        code = compile(ast.fix_missing_locations(ast.Module(
            body=main.body[index:index + 2], type_ignores=[])), str(HERE / 'run_header_hmc.py'), 'exec')

        def check(grid, ranks=2, cli_grid=None):
            ns = dict(args=SimpleNamespace(grid=cli_grid), c=dict(latt_size=[2, 2, 2, 4], grid_size=grid),
                      math=math, MPI=SimpleNamespace(COMM_WORLD=SimpleNamespace(size=ranks)))
            exec(code, ns)
            return ns['grid']

        self.assertEqual(check([1, 1, 1, 2]), [1, 1, 1, 2])
        self.assertIsNone(check(None))
        self.assertEqual(check([3], cli_grid=[1, 1, 1, 2]), [1, 1, 1, 2])
        for grid in ([1, 1, 2], [1, 1, 1, 2, 1], [True, 1, 1, 2], [1., 1, 1, 2],
                     [0, 1, 1, 2], [-1, 1, 1, 2], [1, 1, 1, 3], [1, 1, 1, 1]):
            with self.subTest(grid=grid), self.assertRaisesRegex(ValueError, 'grid must partition'):
                check(grid)

    def test_cg_config_never_reads_mg_file(self):
        source = Mock()
        args = SimpleNamespace(solver="cg", solver_config=source)
        self.assertEqual(RUNNER.solver_config(args), dict(
            plain_precision="double", predictor=dict(enabled=False), multigrid=dict(enabled=False)))
        source.read_text.assert_not_called()

    def test_mg_solver_configuration_requirements(self):
        good = dict(plain_precision="double", predictor=dict(enabled=False),
                    multigrid=dict(enabled=True, force_gradient=True, setup_seed=123))
        args = SimpleNamespace(solver="mg", solver_config=Mock())
        args.solver_config.read_text.return_value = json.dumps(good)
        self.assertEqual(RUNNER.solver_config(args), good)
        for path, value in ((('plain_precision',), 'default'), (('predictor', 'enabled'), True),
                            (('multigrid', 'enabled'), False), (('multigrid', 'force_gradient'), False)):
            with self.subTest(path=path):
                invalid = copy.deepcopy(good)
                set_nested(invalid, path, value)
                args.solver_config.read_text.return_value = json.dumps(invalid)
                with self.assertRaises(ValueError):
                    RUNNER.solver_config(args)
        for key in ('plain_precision', 'multigrid'):
            invalid = copy.deepcopy(good)
            del invalid[key]
            args.solver_config.read_text.return_value = json.dumps(invalid)
            with self.assertRaises(ValueError):
                RUNNER.solver_config(args)

    def test_cli_defaults_and_mg_requires_explicit_config(self):
        argv = ['--config', 'header.json', '--output-dir', 'output']
        args = RUNNER.parse_args(argv)
        self.assertEqual((args.solver, args.arm, args.mode, args.ntraj), ('cg', 'native', 'trajectory', 2))
        for extra in (['--solver', 'mg'], ['--ntraj', '0']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                RUNNER.parse_args(argv + extra)
            self.assertEqual(exc.exception.code, 2)
        args = RUNNER.parse_args(argv + ['--solver', 'mg', '--solver-config', 'mg.json'])
        self.assertEqual(args.solver_config, Path('mg.json'))

    def test_help_works_without_site_packages_or_gpu(self):
        result = subprocess.run([sys.executable, '-B', '-S', str(HERE / 'run_header_hmc.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--config', '--solver', '--solver-config', '--arm', '--mode'):
            self.assertIn(flag, result.stdout)


class GaugeReaderTests(unittest.TestCase):
    def test_reunitarizes_columns_and_leaves_input_unchanged(self):
        matrix = np.array([[2, 1 + 1j, 9], [0, 3j, 7], [0, 0, 5]], dtype=complex)
        batch = np.stack((matrix, matrix * 2))
        before = batch.copy()
        out = SUPPORT.chroma_reunit_columns(batch)
        expected = np.diag([1, 1j, -1j])
        np.testing.assert_allclose(out, np.stack((expected, expected)), atol=1e-15)
        np.testing.assert_array_equal(batch, before)
        self.assertFalse(np.shares_memory(out, batch))
        np.testing.assert_allclose(np.linalg.det(out), 1, atol=1e-15)

    def test_nonfinite_and_degenerate_reunitarization_fail(self):
        for value in (0, float('nan'), float('inf')):
            matrix = np.eye(3, dtype=complex)
            matrix[:, 0] = value
            with self.subTest(value=value), np.errstate(all='ignore'):
                with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                    SUPPORT.chroma_reunit_columns(matrix)

    def reader_stubs(self, precision='D'):
        size = [2, 2, 2, 2]
        width = 8 if precision == 'D' else 4
        docs = {'scidac-private-file-xml': ET.fromstring('<file><dims>2 2 2 2</dims></file>'),
                'scidac-private-record-xml': ET.fromstring(
                    f'<record><precision>{precision}</precision><colors>3</colors><datacount>4</datacount>'
                    f'<typesize>{18 * width}</typesize></record>'),
                'scidac-checksum': ET.fromstring('<checksum><suma>11</suma><sumb>22</sumb></checksum>')}
        lime = Mock()
        lime.loadXML.side_effect = docs.__getitem__
        lime.record.return_value = SimpleNamespace(offset=128)
        matrix = np.array([[2, 1 + 1j, 9], [0, 3j, 7], [0, 0, 5]], dtype=f'>c{2 * width}')
        raw = np.broadcast_to(matrix, (2, 2, 2, 2, 4, 3, 3)).copy()
        read = Mock(return_value=raw)
        checksum = Mock(return_value=(0x11, 0x22))
        comm = SimpleNamespace(allreduce=Mock(side_effect=lambda value, **kw: value))
        gauge = Mock(side_effect=lambda li, data: SimpleNamespace(latt_info=li, data=data))
        li = SimpleNamespace(global_size=size, evenodd=Mock(side_effect=lambda data, parity: data))
        maximum = object()
        modules = {'pyquda_io.lime': module('pyquda_io.lime', Lime=Mock(return_value=lime)),
                   'pyquda_io.mpi_utils': module('pyquda_io.mpi_utils',
                       getSublatticeSize=Mock(return_value=size), readMPIFile=read),
                   'pyquda_io.io_utils': module('pyquda_io.io_utils', checksumSciDAC=checksum),
                   'pyquda.field': module('pyquda.field', LatticeGauge=gauge),
                   'pyquda_comm': module('pyquda_comm', getMPIComm=lambda: comm),
                   'mpi4py': module('mpi4py', MPI=SimpleNamespace(MAX=maximum))}
        return SimpleNamespace(modules=modules, docs=docs, li=li, raw=raw, read=read,
                               checksum=checksum, gauge=gauge, comm=comm, maximum=maximum, lime=lime)

    def test_qio_checksum_precedes_two_column_projections(self):
        for precision, bits in (('D', 64), ('F', 32)):
            with self.subTest(precision=precision):
                stubs = self.reader_stubs(precision)
                before = stubs.raw.copy()
                converted = before.transpose(4, 0, 1, 2, 3, 5, 6).astype('<c16')
                first = SUPPORT.chroma_reunit_columns(converted)
                second = SUPPORT.chroma_reunit_columns(first)
                with patch.dict(sys.modules, stubs.modules), patch.object(
                    SUPPORT, 'chroma_reunit_columns', wraps=SUPPORT.chroma_reunit_columns,
                ) as project:
                    field, info = SUPPORT.read_chroma_gauge('mock.lime', stubs.li)
                self.assertEqual(project.call_count, 2)
                np.testing.assert_array_equal(project.call_args_list[0].args[0], converted)
                np.testing.assert_array_equal(project.call_args_list[1].args[0], first)
                np.testing.assert_array_equal(field.data, second)
                np.testing.assert_array_equal(stubs.raw, before)
                stubs.read.assert_called_once_with('mock.lime', '>c16' if precision == 'D' else '>c8', 128,
                                                   (2, 2, 2, 2, 4, 3, 3), (3, 2, 1, 0))
                stubs.checksum.assert_called_once()
                self.assertEqual(stubs.checksum.call_args.args[0], [2, 2, 2, 2])
                np.testing.assert_array_equal(stubs.checksum.call_args.args[1], before.reshape(16, 36))
                self.assertEqual(stubs.li.evenodd.call_args.args[1], True)
                self.assertEqual(info['input_precision'], bits)
                self.assertTrue(info['reunit'])
                self.assertEqual(info['reunit_passes'], 2)
                self.assertEqual(info['algorithm'], 'Chroma column Gram-Schmidt')
                self.assertFalse(info['bitwise_match_verified'])
                self.assertEqual(info['max_abs_input_change'], float(np.max(np.abs(second - converted))))
                self.assertIs(stubs.comm.allreduce.call_args.kwargs['op'], stubs.maximum)

    def test_invalid_qio_metadata_or_checksum_never_reaches_projection(self):
        cases = [('scidac-private-file-xml', 'dims', '4 2 2 2'),
                 ('scidac-private-record-xml', 'colors', '2'),
                 ('scidac-private-record-xml', 'datacount', '3'),
                 ('scidac-private-record-xml', 'typesize', '1'), ('checksum', None, None)]
        for record, key, value in cases:
            with self.subTest(record=record, key=key):
                stubs = self.reader_stubs()
                if record == 'checksum':
                    stubs.checksum.return_value = (0, 0)
                else:
                    stubs.docs[record].find(key).text = value
                with patch.dict(sys.modules, stubs.modules), patch.object(SUPPORT, 'chroma_reunit_columns') as project:
                    with self.assertRaises(ValueError):
                        SUPPORT.read_chroma_gauge('mock.lime', stubs.li)
                project.assert_not_called()
                stubs.gauge.assert_not_called()


def mg_events(labels=('ratio_h:numer', 'cancel'), scopes=('md', 'temporary')):
    events = []
    for label in labels:
        for scope in scopes:
            events.extend(dict(event='solve', action=label, scope=scope, direction=direction,
                               multigrid_used=True, fallback=False) for direction in ('Mdag', 'M'))
            events.append(dict(event='normal_residual', action=label, scope=scope,
                               true_residual=5e-13, target=1e-12,
                               equation='original_normal', converged=True))
    return events


class MGEvidenceTests(unittest.TestCase):
    labels = ('ratio_h:numer', 'cancel')

    def test_normal_md_retry_accepts_recorded_rebuild_and_final_success(self):
        """setup、伴随或正规方程失败都可重试，但每次须保留完整失败与重建证据。"""
        success = mg_events(labels=(self.labels[0],), scopes=('md',))
        bad = dict(success[-1], true_residual=2e-12, converged=False)
        failure = dict(event='solve_attempt_failure', action=self.labels[0], scope='md',
                       retry=0, fallback=False)
        recovery = [dict(event=event, action=self.labels[0], scope='md')
                    for event in ('rebuild', 'setup')]
        for attempt in ([], success[:1], success[:2], success[:2] + [bad]):
            with self.subTest(attempt=attempt):
                events = mg_events() + attempt + [failure] + recovery + success
                self.assertIsNone(SUPPORT.verify_mg_evidence(events, self.labels, temporary=True))

    def test_each_completed_call_is_checked_even_after_previous_success(self):
        success = mg_events(labels=(self.labels[0],), scopes=('md',))
        # 先前通过的调用不能掩盖后续坏残差或只完成一半的调用。
        incomplete = [success[:1], success[:2],
                      success[:2] + [dict(success[-1], true_residual=2e-12, converged=False)]]
        for attempt in incomplete:
            with self.subTest(attempt=attempt), self.assertRaises(RuntimeError):
                SUPPORT.verify_mg_evidence(mg_events() + attempt, self.labels, temporary=True)
        bad_then_good = success[:2] + [dict(success[-1], true_residual=2e-12, converged=False)] + success
        with self.assertRaises(RuntimeError):
            SUPPORT.verify_mg_evidence(mg_events() + bad_then_good, self.labels, temporary=True)

    def test_unclosed_failure_missing_rebuild_and_second_failure_are_rejected(self):
        label = self.labels[0]
        success = mg_events(labels=(label,), scopes=('md',))
        failure = dict(event='solve_attempt_failure', action=label, scope='md', retry=0)
        rebuild = dict(event='rebuild', action=label, scope='md')
        setup = dict(event='setup', action=label, scope='md')
        tails = ([failure], [failure, rebuild], [failure, rebuild, setup],
                 [failure] + success, [failure, setup] + success,
                 [failure, rebuild] + success,
                 [failure, rebuild, setup] + success[:1] + [dict(failure, retry=1)],
                 [failure, rebuild, setup] + success[:1] + [failure],
                 [dict(failure, retry=1)], [dict(failure, retry=True)])
        for tail in tails:
            with self.subTest(tail=tail), self.assertRaises(RuntimeError):
                SUPPORT.verify_mg_evidence(mg_events() + tail, self.labels, temporary=True)

    def test_silent_plain_cg_cannot_hide_among_successful_mg_calls(self):
        for scope in ('md', 'temporary'):
            for used in (False, None, 1):
                with self.subTest(scope=scope, used=used):
                    event = dict(event='solve', action=self.labels[0], scope=scope,
                                 direction='MdagM', multigrid_used=used, fallback=False)
                    with self.assertRaises(RuntimeError):
                        SUPPORT.verify_mg_evidence(mg_events() + [event], self.labels, temporary=True)

    def test_fg_failure_is_rejected_even_if_followed_by_success(self):
        label = self.labels[0]
        success = mg_events(labels=(label,), scopes=('temporary',))
        bad = dict(success[-1], true_residual=2e-12, converged=False)
        failure = dict(event='solve_attempt_failure', action=label, scope='temporary', retry=0)
        for attempt in ([failure], success[:2] + [bad]):
            with self.subTest(attempt=attempt), self.assertRaises(RuntimeError):
                SUPPORT.verify_mg_evidence(mg_events() + attempt + success, self.labels, temporary=True)

    def test_direction_order_and_original_equation_metadata_are_required(self):
        success = mg_events(labels=(self.labels[0],), scopes=('md',))
        for attempt in ([success[1], success[0], success[2]],
                        [success[0], success[0], success[1], success[2]],
                        [success[2]], [success[0], success[2]]):
            with self.subTest(attempt=attempt), self.assertRaises(RuntimeError):
                SUPPORT.verify_mg_evidence(mg_events() + attempt, self.labels, temporary=True)
        for key, value in (('equation', 'original_adjoint'), ('equation', None),
                           ('converged', None), ('converged', 1), ('converged', False)):
            with self.subTest(key=key, value=value):
                events = mg_events()
                next(e for e in events if e['event'] == 'normal_residual')[key] = value
                with self.assertRaises(RuntimeError):
                    SUPPORT.verify_mg_evidence(events, self.labels, temporary=True)

    def test_fg_update_checks_only_update_fields_not_unrelated_pool_events(self):
        update = dict(event='update', action=self.labels[0], scope='temporary',
                      refresh_iterations=0, basis_frozen=True)
        self.assertIsNone(SUPPORT.verify_mg_evidence(mg_events() + [update], self.labels, temporary=True))
        for change in (dict(refresh_iterations=1), dict(basis_frozen=False), dict(basis_frozen=None)):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                SUPPORT.verify_mg_evidence(mg_events() + [dict(update, **change)], self.labels, temporary=True)
        # setup/rebuild 事件本来没有 update 字段，普通 MD 不应因此被误拒绝。
        pool = [dict(event=kind, action=self.labels[0], scope='md') for kind in ('rebuild', 'setup')]
        self.assertIsNone(SUPPORT.verify_mg_evidence(pool + mg_events(), self.labels, temporary=True))

    def test_multiple_calls_can_interleave_distinct_action_evidence(self):
        left = mg_events(labels=(self.labels[0],), scopes=('md',))
        right = mg_events(labels=(self.labels[1],), scopes=('md',))
        interleaved = [event for pair in zip(left, right) for event in pair]
        self.assertIsNone(SUPPORT.verify_mg_evidence(
            interleaved + interleaved + mg_events(scopes=('temporary',)), self.labels, temporary=True))

    def test_per_action_md_targets_accept_header_tolerances_and_stricter_gate(self):
        targets = {'ratio_h:numer': 1e-8, 'cancel': 1e-10}
        for strict_gate in (False, True):
            with self.subTest(strict_gate=strict_gate):
                events = mg_events()
                for event in events:
                    if event['event'] == 'normal_residual':
                        target = 1e-12 if strict_gate and event['scope'] == 'md' else targets[event['action']]
                        event.update(target=target, true_residual=target * .5)
                self.assertIsNone(SUPPORT.verify_mg_evidence(
                    events, self.labels, temporary=True, expected_md_tols=targets))

    def test_per_action_md_targets_reject_looser_wrong_action_and_failed_residuals(self):
        targets = {'ratio_h:numer': 1e-8, 'cancel': 1e-10}
        for label in self.labels:
            for scope in ('md', 'temporary'):
                for target, residual in ((targets[label] * 2, targets[label] * .5),
                                         (targets[label], targets[label] * 2), (targets[label], -1.)):
                    with self.subTest(label=label, scope=scope, target=target, residual=residual):
                        events = mg_events()
                        event = next(e for e in events if e['event'] == 'normal_residual'
                                     and e['action'] == label and e['scope'] == scope)
                        event.update(target=target, true_residual=residual)
                        with self.assertRaisesRegex(RuntimeError, 'missing real MG'):
                            SUPPORT.verify_mg_evidence(events, self.labels, temporary=True,
                                                       expected_md_tols=targets)
        events = mg_events()
        for event in events:
            if event['event'] == 'normal_residual' and event['action'] == 'cancel':
                event.update(target=targets['ratio_h:numer'], true_residual=1e-11)
        with self.assertRaisesRegex(RuntimeError, 'cancel'):
            SUPPORT.verify_mg_evidence(events, self.labels, temporary=True, expected_md_tols=targets)

    def test_expected_md_targets_require_every_action_and_valid_finite_values(self):
        for value in (None, True, '1e-8', 0, -1e-8, 1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SUPPORT.verify_mg_evidence(mg_events(), self.labels, temporary=True,
                                           expected_md_tols={self.labels[0]: value, self.labels[1]: 1e-10})
        with self.assertRaisesRegex(ValueError, 'cancel'):
            SUPPORT.verify_mg_evidence(mg_events(), self.labels, temporary=True,
                                       expected_md_tols={self.labels[0]: 1e-8})

    def test_accepts_both_event_schemas_and_requires_temporary_when_requested(self):
        events = mg_events()
        events += [dict(event='solve', scope=scope, multigrid_used=False) for scope in ('sample', 'action')]
        events.append(dict(event='setup', scope='md'))
        self.assertIsNone(SUPPORT.verify_mg_evidence(events, self.labels, temporary=True))
        old_schema = [dict(type=e['event'], **{k: v for k, v in e.items() if k != 'event'}) for e in events]
        self.assertIsNone(SUPPORT.verify_mg_evidence({'events': old_schema}, self.labels, temporary=True))
        md_only = mg_events(scopes=('md',))
        self.assertIsNone(SUPPORT.verify_mg_evidence(md_only, self.labels, temporary=False))
        with self.assertRaisesRegex(RuntimeError, 'temporary'):
            SUPPORT.verify_mg_evidence(md_only, self.labels, temporary=True)

    def test_temporary_basis_mutation_fallback_and_non_cg_refresh_are_rejected(self):
        forbidden = [dict(event=kind, scope='temporary') for kind in ('setup', 'refresh', 'rebuild')]
        forbidden += [dict(event='solve', scope='md', fallback=True),
                      dict(event='solve', scope='temporary', fallback='cg')]
        forbidden += [dict(event='solve', scope=scope, multigrid_used=True) for scope in ('sample', 'action')]
        for event in forbidden:
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                SUPPORT.verify_mg_evidence(mg_events() + [event], self.labels, temporary=True)

    def test_each_action_scope_direction_and_normal_residual_are_required(self):
        for label in self.labels:
            for scope in ('md', 'temporary'):
                for missing in ('Mdag', 'M', 'normal_residual', 'all'):
                    with self.subTest(label=label, scope=scope, missing=missing):
                        events = [e for e in mg_events() if not (
                            e['action'] == label and e['scope'] == scope and (
                                missing == 'all' or e.get('direction') == missing or e['event'] == missing))]
                        with self.assertRaisesRegex(RuntimeError, 'missing real MG'):
                            SUPPORT.verify_mg_evidence(events, self.labels, temporary=True)
        with self.assertRaises(RuntimeError):
            SUPPORT.verify_mg_evidence([], self.labels, temporary=False)
        with self.assertRaises(RuntimeError):
            SUPPORT.verify_mg_evidence(mg_events(labels=('other',)), self.labels, temporary=True)

    def test_enabled_setup_or_plain_cg_does_not_count_as_mg(self):
        events = mg_events()
        for e in events:
            if e['event'] == 'solve':
                e['multigrid_used'] = False
        events.append(dict(event='setup', scope='md', enabled=True))
        with self.assertRaisesRegex(RuntimeError, 'missing real MG'):
            SUPPORT.verify_mg_evidence(events, self.labels, temporary=True)

    def test_invalid_original_residuals_and_targets_are_rejected(self):
        cases = [('true_residual', None), ('true_residual', '1e-13'), ('true_residual', 2e-12),
                 ('true_residual', float('nan')), ('true_residual', float('inf')),
                 ('target', 0), ('target', -1e-12), ('target', 2e-12),
                 ('target', float('nan')), ('target', float('inf'))]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                events = mg_events()
                residual = next(e for e in events if e['event'] == 'normal_residual')
                residual[key] = value
                with self.assertRaisesRegex(RuntimeError, 'missing real MG'):
                    SUPPORT.verify_mg_evidence(events, self.labels, temporary=True)
        events = mg_events()
        residual = next(e for e in events if e['event'] == 'normal_residual')
        del residual['target']
        with self.assertRaises(RuntimeError):
            SUPPORT.verify_mg_evidence(events, self.labels, temporary=True)


if __name__ == '__main__':
    unittest.main()
