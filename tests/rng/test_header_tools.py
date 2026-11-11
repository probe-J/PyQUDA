"""synthetic XML/JSONL 的纯 CPU unittest，不需要 primary、MPI 或 GPU。

运行：python3 -B -m unittest discover -s tests/rng -p test_header_tools.py -v
测试在系统临时目录生成最小输入/输出，覆盖物理保留、显式 copy、拒绝未知模型、
数量/有限值/RNG/接受状态和比较器退出码；不复制历史日志或模板产物。
"""

import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

import compare_hmc_results as compare
import extract_chroma_header as extract


def add(parent, tag, text=None):
    node = ET.SubElement(parent, tag)
    if text is not None:
        node.text = str(text)
    return node


def seed(parent, words):
    node = add(add(parent, "RNG"), "Seed")
    for word in words:
        add(node, "elem", word)


def fermion(parent, mass):
    node = add(parent, "FermionAction")
    for k, v in dict(FermAct="SEOPREC_CLOVER", Mass=mass, clovCoeff=1.1).items():
        add(node, k, v)
    state = add(node, "FermState")
    for k, v in dict(Name="STOUT_FERM_STATE", rho=.12, n_smear=1, orthog_dir=-1).items():
        add(state, k, v)
    bc = add(state, "FermionBC")
    add(bc, "FermBC", "SIMPLE_FERMBC")
    add(bc, "boundary", "1 1 1 -1")


def inverter(parent, tolerance="1e-12", cap=200):
    node = add(parent, "InvertParam")
    add(node, "invType", "QUDA_MULTIGRID_CLOVER_INVERTER")
    add(node, "RsdTarget", tolerance)
    add(node, "MaxIter", cap)


def synthetic_header():
    """一段 scalar chain、两个 logdet、一个 rational；名字刻意不依赖 HMC/MD 后缀。"""
    root = ET.Element("Params")
    mc = add(root, "MCControl")
    cfg = add(mc, "Cfg")
    for k, v in dict(cfg_type="SZINQIO", cfg_file="old.lime", reunit="true").items():
        add(cfg, k, v)
    seed(mc, [1, 2, 3, 4])
    for k, v in dict(StartUpdateNum=8, NWarmUpUpdates=2, NProductionUpdates=100,
                     NUpdatesThisRun=40, SaveInterval=10, SavePrefix="old").items():
        add(mc, k, v)
    add(add(add(mc, "InlineMeasurements"), "elem"), "Name", "PLAQUETTE")
    trj = add(root, "HMCTrj")
    add(trj, "nrow", "4 4 4 8")
    monos = add(trj, "Monomials")
    def monomial(mid, role):
        m = add(monos, "elem")
        add(m, "Name", next(k for k, v in extract.ROLES.items() if v == role))
        add(add(m, "NamedObject"), "monomial_id", mid)
        return m
    for mid, mass, nf in (("light_log", -.3, 2), ("strange_log", -.1, 1)):
        m = monomial(mid, "logdet")
        fermion(m, mass)
        add(m, "num_flavors", nf)
    m = monomial("rational", "strange")
    add(m, "num_pf", 1)
    action = add(m, "Action")
    fermion(action, -.1)
    for kind in ("ActionApprox", "ForceApprox"):
        approx = add(action, kind)
        rat = add(approx, "RationalApprox")
        add(rat, "ratApproxType", "READ_COEFFS")
        for coeff in ("PFECoeffs", "IPFECoeffs"):
            c = add(rat, coeff)
            for k, v in dict(norm=.5, res=".2 .3", pole=".1 .4").items():
                add(c, k, v)
        inverter(approx, "1e-9 1e-9", 5000)
    for mid, tol in (("ratio_h", "1e-12"), ("different_ratio_md", "1e-10")):
        m = monomial(mid, "hasenbusch_ratio")
        for tag, mass in (("Action", -.3), ("PrecAction", -.2)):
            a = add(m, tag)
            fermion(a, mass)
            inverter(a, tol)
    for mid in ("cancel_h", "different_cancel_md"):
        m = monomial(mid, "hasenbusch_cancel")
        fermion(m, -.2)
        inverter(m)
    m = monomial("gauge", "gauge")
    ga = add(m, "GaugeAction")
    for k, v in dict(Name="LW_TREE_GAUGEACT", beta=6.1, u0=.9).items():
        add(ga, k, v)
    add(add(ga, "GaugeBC"), "Name", "PERIODIC_GAUGEBC")
    h = add(add(trj, "Hamiltonian"), "monomial_ids")
    for mid in ("strange_log", "rational", "light_log", "ratio_h", "cancel_h", "gauge"):
        add(h, "elem", mid)
    md = add(trj, "MDIntegrator")
    add(md, "tau0", 1)
    copies = add(md, "copyList")
    for source, dest in (("ratio_h", "different_ratio_md"), ("cancel_h", "different_cancel_md")):
        pair = add(copies, "elem")
        add(pair, "copyFrom", source)
        add(pair, "copyTo", dest)
    outer = add(md, "Integrator")
    inner = add(outer, "SubIntegrator")
    for node, steps, ids in ((outer, 7, ("light_log", "strange_log", "rational", "different_ratio_md")),
                             (inner, 2, ("different_cancel_md", "gauge"))):
        add(node, "Name", "LCM_STS_FORCE_GRAD")
        add(node, "n_steps", steps)
        for mid in ids:
            add(node.find("monomial_ids") if node.find("monomial_ids") is not None else add(node, "monomial_ids"), "elem", mid)
    return root


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.root = synthetic_header()

    def convert(self, **kwargs):
        return extract.convert(ET.tostring(self.root), "gauge.lime", "run", **kwargs)

    def test_preservation_and_schema(self):
        before = ET.tostring(self.root)
        out, config, audit = self.convert()
        self.assertEqual(extract.signature(self.root.find("HMCTrj")), extract.signature(out.find("HMCTrj")))
        self.assertEqual(ET.tostring(self.root), before)
        self.assertEqual(config["hamiltonian_order"], ["strange_log", "rational", "light_log", "ratio_h", "cancel_h", "gauge"])
        self.assertEqual(config["copy_list"][0], dict(copyFrom="ratio_h", copyTo="different_ratio_md"))
        self.assertEqual(config["integrator"]["outer_type"], "LCM_STS_FORCE_GRAD")
        self.assertEqual(config["integrator"]["inner_type"], "LCM_STS_FORCE_GRAD")
        self.assertEqual(config["sampling_order"][4:], ["rational", "ratio_h", "cancel_h"])
        self.assertEqual(config["boundary"], [1, 1, 1, -1])
        self.assertEqual(config["masses"], [-.3, -.2])
        self.assertEqual(config["clover_csw_light"], 1.1)
        self.assertEqual(config["integrator"]["tau0"], 1)
        self.assertEqual(config["seed_packed"], sum(x << (12*i) for i, x in enumerate([1, 2, 3, 4])))
        self.assertEqual(out.findtext("MCControl/NProductionUpdates"), "8")
        self.assertEqual(out.findtext("MCControl/NUpdatesThisRun"), "2")
        self.assertEqual(out.findtext("MCControl/SavePrefix"), "run/chroma")
        self.assertEqual(audit["cancel_maxiter_overrides"], [])
        self.assertTrue(all(r["maxiter"] == 200 for r in config["monomials"] if r["role"] == "hasenbusch_cancel"))
        self.assertEqual(config["monomials"][2]["solver_blocks"][0]["tolerance"], [1e-9, 1e-9])
        self.assertTrue(all(e.text == "CG_INVERTER" for e in out.findall(".//invType")))

    def test_explicit_override_and_ntraj(self):
        out, config, audit = self.convert(cancel_maxiter=5000, ntraj=3)
        self.assertEqual(len(audit["cancel_maxiter_overrides"]), 2)
        self.assertEqual(out.findtext("MCControl/NProductionUpdates"), "9")
        self.assertEqual(out.findtext("MCControl/NUpdatesThisRun"), "3")
        self.assertTrue(all(r["maxiter"] == 5000 for r in config["monomials"] if r["role"] == "hasenbusch_cancel"))

    def test_reject_unsupported_physics(self):
        paths = {
            "HMCTrj/Monomials/elem/Name": "UNKNOWN_CANCEL_LIKE_MONOMIAL",
            ".//FermAct": "TWISTED_CLOVER",
            ".//boundary": "-1 1 1 -1",
            ".//clovCoeff": "nan",
            ".//num_pf": "2",
            ".//orthog_dir": "1",
            "HMCTrj/MDIntegrator/Integrator/Name": "LEAPFROG",
            "HMCTrj/MDIntegrator/Integrator/n_steps": "0",
            "MCControl/Cfg/reunit": "false",
            "MCControl/StartUpdateNum": "1",
            "MCControl/RNG/Seed/elem": "4096",
            ".//ratApproxType": "GENERATE_COEFFS",
            ".//RsdTarget": "1e-9 1e-8",
            ".//PFECoeffs/pole": ".1",
        }
        original = copy.deepcopy(self.root)
        for path, value in paths.items():
            with self.subTest(path=path):
                self.root = copy.deepcopy(original)
                self.root.find(path).text = value
                with self.assertRaises(ValueError):
                    self.convert()

    def test_unknown_physical_field(self):
        add(self.root.find(".//FermionAction"), "TwistedMass", .02)
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.convert()

    def test_unknown_nested_action_field(self):
        add(self.root.find(".//PrecAction"), "ShiftedMass", .02)
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            self.convert()

    def test_already_canonical_without_copy_list(self):
        md = self.root.find("HMCTrj/MDIntegrator")
        md.remove(md.find("copyList"))
        aliases = {"different_ratio_md": "ratio_h", "different_cancel_md": "cancel_h"}
        for node in md.findall(".//monomial_ids/elem"):
            node.text = aliases.get(node.text, node.text)
        monos = self.root.find("HMCTrj/Monomials")
        for node in list(monos):
            if node.findtext("NamedObject/monomial_id") in aliases:
                monos.remove(node)
        _, config, _ = self.convert()
        self.assertEqual(config["copy_list"], [])
        self.assertEqual(len(config["monomials"]), 6)

    def test_seed_fourth_word_and_copy_cap(self):
        self.root.findall("MCControl/RNG/Seed/elem")[3].text = "2048"
        with self.assertRaisesRegex(ValueError, "47-bit"):
            self.convert()
        self.root = synthetic_header()
        self.root.findall(".//PrecAction/InvertParam/MaxIter")[1].text = "201"
        with self.assertRaises(ValueError):
            self.convert()

    def test_copy_required_and_physics_checked(self):
        md = self.root.find("HMCTrj/MDIntegrator")
        md.remove(md.find("copyList"))
        with self.assertRaisesRegex(ValueError, "explicit copyList"):
            self.convert()
        self.root = synthetic_header()
        self.root.find("HMCTrj/MDIntegrator/copyList/elem/copyTo").text = "gauge"
        with self.assertRaises(ValueError):
            self.convert()
        self.root = synthetic_header()
        ratios = self.root.findall(".//PrecAction/FermionAction/Mass")
        ratios[1].text = "-.19"
        with self.assertRaisesRegex(ValueError, "physics mismatch"):
            self.convert()

    def test_duplicate_and_deeper_integrator(self):
        add(self.root.find("HMCTrj/Hamiltonian/monomial_ids"), "elem", "gauge")
        with self.assertRaises(ValueError):
            self.convert()
        self.root = synthetic_header()
        add(self.root.find(".//SubIntegrator"), "SubIntegrator")
        with self.assertRaises(ValueError):
            self.convert()

    def test_bad_arguments(self):
        for kw in (dict(ntraj=0), dict(ntraj=True), dict(cancel_maxiter=0)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.convert(**kw)

    def test_cli_and_lime(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = ET.tostring(self.root)
            header, lime, output = tmp / "source.xml", tmp / "gauge.lime", tmp / "out"
            header.write_bytes(raw)
            block = struct.pack(">IHHQ128s", 0x456789AB, 1, 0, len(raw), b"scidac-record-xml")
            lime.write_bytes(block + raw)
            self.assertEqual(extract.read_header(lime), raw)
            self.assertEqual(extract.main(["--gauge", str(lime), "--header", str(header), "--output-dir", str(output)]), 0)
            self.assertEqual({p.name for p in output.iterdir()}, {"header.xml", "input.xml", "config.json", "audit.json"})
            self.assertEqual((output / "header.xml").read_bytes(), raw)
            lime.write_bytes(block + raw[:-1])
            with self.assertRaises(ValueError):
                extract.read_header(lime)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.chroma, self.pyquda = self.base / "output.xml", self.base / "rank0.jsonl"
        self.root = ET.Element("chroma")
        seed(add(self.root, "MCControl"), [1, 0, 0, 0])
        updates = add(self.root, "MCUpdates")
        self.qrows = []
        for i in range(2):
            update = add(add(updates, "elem"), "Update")
            add(update, "update_no", 11 + i)
            trj = add(update, "HMCTrajectory")
            add(trj, "DeltaH", .1 * (i+1))
            add(trj, "AcceptP", "true" if i == 0 else "false")
            add(add(update, "Plaquette"), "w_plaq", .6)
            seed(update, [i+2, 0, 0, 0])
            self.qrows.append(dict(type="trajectory", trajectory=i+1, delta_H=.1 * (i+1),
                                   accept=i == 0, plaquette=[.6, .61, .59],
                                   seed_before=i+1, seed_after_refresh=100+i, seed_after=i+2))

    def run_compare(self, **kw):
        ET.ElementTree(self.root).write(self.chroma)
        self.pyquda.write_text("\n".join(json.dumps(r) for r in self.qrows) + "\n", encoding="utf-8")
        return compare.compare(self.chroma, self.pyquda, 2, **kw)

    def test_matching(self):
        result = self.run_compare()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["initial_seed_checked"])
        self.assertEqual(len(result["paired"]), 2)

    def inline_plaquettes(self, with_update_no=True):
        """模拟实际 output.xml 的正式 InlineObservables 布局与测量轨迹号。"""
        nodes = []
        for i, update in enumerate(self.root.findall("MCUpdates/elem/Update")):
            update.find("update_no").text = str(10002 + i)
            node = update.find("Plaquette")
            update.remove(node)
            add(add(update, "InlineObservables"), "elem").append(node)
            if with_update_no:
                add(node, "update_no", 10002 + i)
            nodes.append(node)
        return nodes

    def test_runtime_input_seed_excludes_configuration_history(self):
        control = self.root.find("MCControl")
        self.root.remove(control)
        add(add(self.root, "Input"), "Params").append(control)
        history = add(add(self.root, "Config_info"), "Params")
        seed(add(history, "MCControl"), [99, 0, 0, 0])
        result = self.run_compare()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["initial_seed_checked"])

    def test_duplicate_runtime_seed_rejected(self):
        seed(add(add(add(self.root, "Input"), "Params"), "MCControl"), [1, 0, 0, 0])
        result = self.run_compare()
        self.assertFalse(result["passed"])
        self.assertIn("ambiguous Chroma initial seed", result["issues"][0])

    def test_inline_formal_plaquette(self):
        self.inline_plaquettes()
        for update in self.root.findall("MCUpdates/elem/Update"):
            # 干扰值与正式值不同，证明调试量不会参与比较或制造歧义。
            add(add(add(update, "SmearDebug"), "Plaquette"), "w_plaq", .9)
            elem = add(update.find("InlineObservables"), "elem")
            add(add(add(elem, "SmearDebug"), "Plaquette"), "w_plaq", .8)
        result = self.run_compare()
        self.assertTrue(result["passed"], result)
        self.assertEqual([r["chroma"]["update_no"] for r in result["paired"]], [10002, 10003])

    def test_inline_formal_plaquette_without_update_no(self):
        self.inline_plaquettes(with_update_no=False)
        self.assertTrue(self.run_compare()["passed"])

    def test_inline_duplicate_formal_plaquettes_rejected(self):
        node = self.inline_plaquettes()[0]
        update = self.root.find("MCUpdates/elem/Update")
        original = copy.deepcopy(update)
        for path in ("InlineObservables/elem", "InlineObservables", ".", "Observables"):
            with self.subTest(path=path):
                update.clear()
                update.extend(copy.deepcopy(list(original)))
                duplicate = copy.deepcopy(node)
                if path == "InlineObservables":
                    parent = add(update.find(path), "elem")
                elif path == "Observables":
                    parent = add(update, path)
                else:
                    parent = update.find(path)
                parent.append(duplicate)
                result = self.run_compare()
                self.assertFalse(result["passed"])
                self.assertIn("ambiguous Chroma formal plaquette", result["issues"][0])

    def test_inline_plaquette_update_no_rejected(self):
        node = self.inline_plaquettes()[0]
        for value in ("10001", "", "bad"):
            with self.subTest(value=value):
                node.find("update_no").text = value
                self.assertFalse(self.run_compare()["passed"])
        node.find("update_no").text = "10002"
        add(node, "update_no", 10002)
        self.assertFalse(self.run_compare()["passed"])

    def test_inline_diagnostic_plaquette_does_not_pass(self):
        node = self.inline_plaquettes()[0]
        elem = self.root.find("MCUpdates/elem/Update/InlineObservables/elem")
        elem.remove(node)
        add(elem, "SmearDebug").append(node)
        self.assertFalse(self.run_compare()["passed"])

    def test_observables_formal_plaquette(self):
        for update in self.root.findall("MCUpdates/elem/Update"):
            node = update.find("Plaquette")
            update.remove(node)
            add(update, "Observables").append(node)
        self.assertTrue(self.run_compare()["passed"])

    def test_missing_extra_duplicate(self):
        original = copy.deepcopy(self.qrows)
        for rows in (original[:1], original + [original[0]], [original[0], original[0]]):
            with self.subTest(rows=rows):
                self.qrows = copy.deepcopy(rows)
                self.assertFalse(self.run_compare()["passed"])

    def test_mismatches_nonfinite_missing(self):
        original = copy.deepcopy(self.qrows)
        for key, value in (("delta_H", .5), ("plaquette", [.7, .7, .7]),
                           ("accept", False), ("accept", "false"), ("seed_after", 7),
                           ("seed_before", 8), ("seed_after_refresh", None),
                           ("delta_H", float("nan")), ("plaquette", [.6, float("inf"), .6]),
                           ("energy", {"H": float("inf")})):
            with self.subTest(key=key, value=value):
                self.qrows = copy.deepcopy(original)
                self.qrows[0][key] = value
                self.assertFalse(self.run_compare()["passed"])

    def test_seed_continuity_and_failure_record(self):
        self.qrows[1]["seed_before"] = 8
        self.assertFalse(self.run_compare()["passed"])
        self.qrows[1]["seed_before"] = 2
        self.qrows.append(dict(type="failure", error="nonconvergence"))
        self.assertFalse(self.run_compare()["passed"])

    def test_explicit_restart_and_missing_seed(self):
        restarts = []
        for i, u in enumerate(self.root.findall(".//Update")):
            u.remove(u.find("RNG"))
            restart = ET.Element("Params")
            seed(add(restart, "MCControl"), [i+2, 0, 0, 0])
            path = self.base / f"explicit_{i}.xml"
            ET.ElementTree(restart).write(path)
            restarts.append(path)
        self.assertFalse(self.run_compare()["passed"])
        self.assertTrue(self.run_compare(restarts=restarts)["passed"])
        self.assertFalse(self.run_compare(restarts=restarts[:1])["passed"])

    def test_diagnostic_plaquette_does_not_pass(self):
        update = self.root.find(".//Update")
        update.remove(update.find("Plaquette"))
        add(add(add(update, "SmearDebug"), "Plaquette"), "w_plaq", .6)
        self.assertFalse(self.run_compare()["passed"])

    def test_incomplete_chroma_rejected(self):
        trj = self.root.find(".//HMCTrajectory")
        trj.remove(trj.find("DeltaH"))
        self.assertFalse(self.run_compare()["passed"])

    def test_threshold_and_cli_exit(self):
        self.qrows[0]["delta_H"] += 2e-7
        self.assertFalse(self.run_compare()["passed"])
        self.assertTrue(self.run_compare(dh_atol=3e-7)["passed"])
        output = self.base / "report.json"
        args = ["--chroma", str(self.chroma), "--pyquda", str(self.pyquda), "--output", str(output), "--ntraj", "2"]
        self.assertEqual(compare.main(args), 1)
        self.assertFalse(json.loads(output.read_text())["passed"])
        self.assertEqual(compare.main(args + ["--dh-atol", "3e-7"]), 0)


if __name__ == "__main__":
    unittest.main()
