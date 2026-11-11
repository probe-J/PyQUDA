#!/usr/bin/env python3
"""从本地 LIME/XML 提取受支持的 scalar Clover 两层 STSForceGrad 输入。

仅用标准库，不初始化 MPI/GPU。CLI 必须给出 --gauge 和 --output-dir；
可用 --header 指定独立头文件。输出 header.xml、input.xml、config.json、
audit.json。默认保留 cancel 上限，只有 --cancel-maxiter 才显式覆盖。
convert 不写文件，返回 (input XML 根节点, config, audit)。
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET


ROLES = {
    "N_FLAVOR_LOGDET_DIAG_FERM_MONOMIAL": "logdet",
    "ONE_FLAVOR_SEOPREC_CONSTDET_FERM_RAT_MONOMIAL": "strange",
    "TWO_FLAVOR_SEOPREC_CONSTDET_RATIO_CONV_CONV_FERM_MONOMIAL": "hasenbusch_ratio",
    "TWO_FLAVOR_SEOPREC_CONSTDET_FERM_MONOMIAL": "hasenbusch_cancel",
    "GAUGE_MONOMIAL": "gauge",
}


def required(node, path):
    value = None if node is None else node.findtext(path)
    if value is None or not value.strip():
        raise ValueError("missing XML field: " + path)
    return value.strip()


def numbers(node):
    if node is None:
        raise ValueError("missing numeric XML field")
    values = [float(x) for x in " ".join(node.itertext()).split()]
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("empty/non-finite numeric XML field: " + node.tag)
    return values


def number(node, path):
    values = numbers(None if node is None else node.find(path))
    if len(values) != 1:
        raise ValueError("expected scalar: " + path)
    return values[0]


def integer(node, path, minimum=0):
    value = int(required(node, path))
    if value < minimum:
        raise ValueError("integer below minimum: " + path)
    return value


def boolean(node, path):
    value = required(node, path).lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError("invalid boolean: " + path)
    return value in ("true", "1")


def only(node, allowed):
    """物理字段白名单：保留原 XML 不能代替 runner 对字段的实际支持。"""
    if node is None:
        raise ValueError("missing required XML structure")
    tags = [c.tag for c in node]
    if len(tags) != len(set(tags)) or set(tags) - set(allowed):
        raise ValueError("duplicate/unsupported fields in " + node.tag + ": " + str(tags))


def xml(node):
    return ET.tostring(node, encoding="unicode")


def read_header(path):
    """只读扫描 LIME 记录，跳过规范场二进制数据；XML 上限 16 MiB。"""
    if Path(path).suffix.lower() == ".xml":
        return Path(path).read_bytes().rstrip(b"\0")
    with open(path, "rb") as stream:
        while True:
            block = stream.read(144)
            if len(block) != 144:
                raise ValueError("missing/truncated LIME record header")
            magic, version, _, length, kind = struct.unpack(">IHHQ128s", block)
            if magic != 0x456789AB or version != 1:
                raise ValueError("invalid LIME magic/version")
            if kind.rstrip(b"\0") == b"scidac-record-xml":
                if length > 16 * 1024 * 1024:
                    raise ValueError("LIME XML exceeds 16 MiB")
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError("truncated LIME XML")
                return raw.rstrip(b"\0")
            stream.seek((length + 7) // 8 * 8, 1)


def signature(node):
    """只忽略求解器/初猜；用于保证转换未改变物理和积分器。"""
    if node.tag.lower() in ("invparam", "invertparam") or "predictor" in node.tag.lower():
        return None
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or "").strip(),
            tuple(s for child in node if (s := signature(child)) is not None))


def fermion_info(ferm):
    only(ferm, {"FermAct", "Mass", "clovCoeff", "AnisoParam", "FermState"})
    if required(ferm, "FermAct") != "SEOPREC_CLOVER":
        raise ValueError("only scalar SEOPREC_CLOVER is supported")
    mass, csw = number(ferm, "Mass"), number(ferm, "clovCoeff")
    if mass <= -4 or csw <= 0:
        raise ValueError("invalid Clover mass/coefficient")
    aniso = ferm.find("AnisoParam")
    if aniso is not None:
        only(aniso, {"anisoP", "t_dir", "xi_0", "nu"})
        if (boolean(aniso, "anisoP") or integer(aniso, "t_dir") != 3
                or number(aniso, "xi_0") != 1 or number(aniso, "nu") != 1):
            raise ValueError("anisotropic Clover is unsupported")
    state = ferm.find("FermState")
    if state is None:
        raise ValueError("missing FermState")
    only(state, {"Name", "rho", "n_smear", "orthog_dir", "FermionBC"})
    if required(state, "Name") != "STOUT_FERM_STATE":
        raise ValueError("only STOUT_FERM_STATE is supported")
    stout = dict(n_smear=integer(state, "n_smear", 1), rho=number(state, "rho"),
                 dir_ignore=int(required(state, "orthog_dir")))
    if stout["rho"] < 0 or stout["dir_ignore"] not in (-1, 3, 4):
        raise ValueError("unsupported stout directions/strength")
    bc = state.find("FermionBC")
    if bc is None:
        raise ValueError("missing FermionBC")
    only(bc, {"FermBC", "boundary"})
    boundary = [int(x) for x in required(bc, "boundary").split()]
    if required(bc, "FermBC") != "SIMPLE_FERMBC" or boundary[:3] != [1, 1, 1] or len(boundary) != 4 or boundary[3] not in (-1, 1):
        raise ValueError("only periodic space and temporal +/-1 FermionBC supported")
    return dict(mass=mass, clover_csw=csw, stout=stout, boundary=boundary)


def rational(monomial):
    result = {}
    for role, approx, coeff in (("action", "ActionApprox", "PFECoeffs"),
                                ("sample", "ActionApprox", "IPFECoeffs"),
                                ("force", "ForceApprox", "PFECoeffs")):
        candidates = monomial.findall(".//" + approx + "/RationalApprox")
        if len(candidates) != 1 or required(candidates[0], "ratApproxType") != "READ_COEFFS":
            raise ValueError("missing/unsupported rational approximation: " + role)
        only(candidates[0], {"ratApproxType", "ApproxInfo", "PFECoeffs", "IPFECoeffs"})
        c = candidates[0].find(coeff)
        if c is None:
            raise ValueError("missing rational coefficients: " + role)
        only(c, {"norm", "res", "pole"})
        residues, poles = numbers(None if c is None else c.find("res")), numbers(None if c is None else c.find("pole"))
        if len(residues) != len(poles) or min(poles) < 0:
            raise ValueError("invalid rational residues/poles")
        info = candidates[0].find("ApproxInfo")
        if info is not None:
            only(info, {"lowerMin", "upperMax", "numPower", "denPower", "degree", "error"})
            expected_power = 2 if role == "force" else 4
            if (int(required(info, "numPower")) != -1 or int(required(info, "denPower")) != expected_power
                    or integer(info, "degree", 1) != len(poles)
                    or not 0 < number(info, "lowerMin") < number(info, "upperMax")
                    or number(info, "error") < 0):
                raise ValueError("unsupported one-flavor rational exponent/degree/domain")
        result.update({"norm_" + role: number(c, "norm"),
                       "residue_" + role: residues, "offset_" + role: poles})
    return result


def solver_info(monomial):
    blocks = []
    for inv in monomial.iter():
        if inv.tag.lower() not in ("invparam", "invertparam"):
            continue
        target = inv.find("RsdTarget")
        if target is None:
            target = inv.find("RsdCG")
        values = numbers(target)
        if not all(0 < t < 1 for t in values):
            raise ValueError("solver tolerance must lie in (0,1)")
        if len(set(values)) != 1:
            raise ValueError("unequal per-pole tolerances unsupported by current runner")
        if inv.find("RsdCG") is not None and inv.find("RsdTarget") is not None:
            if set(numbers(inv.find("RsdCG"))) != set(values):
                raise ValueError("conflicting RsdCG/RsdTarget")
        cap = inv.findtext("MaxCG") or inv.findtext("MaxIter")
        if cap is None or int(cap) <= 0:
            raise ValueError("missing/invalid iteration cap")
        if inv.find("MaxCG") is not None and inv.find("MaxIter") is not None:
            if int(inv.findtext("MaxCG")) != int(inv.findtext("MaxIter")):
                raise ValueError("conflicting MaxCG/MaxIter")
        blocks.append(dict(inv_type=required(inv, "invType"), tolerance=values,
                           tol=values[0], maxiter=int(cap), xml=xml(inv)))
    return blocks


def set_text(node, key, value):
    child = node.find(key)
    if child is None:
        child = ET.SubElement(node, key)
    child.text = str(value)


def convert(raw, gauge, output_dir, cancel_maxiter=None, ntraj=2):
    """纯转换；严格保留 scalar 质量、残差、积分器和 copy 语义。"""
    if type(ntraj) is not int or ntraj <= 0:
        raise ValueError("ntraj must be a positive integer")
    if cancel_maxiter is not None and (type(cancel_maxiter) is not int or cancel_maxiter <= 0):
        raise ValueError("cancel_maxiter must be a positive integer")
    if not str(gauge) or not str(output_dir):
        raise ValueError("gauge/output_dir must be explicit paths")
    original = ET.fromstring(raw)
    if original.tag != "Params":
        raise ValueError("expected Params root")
    trj, mc = original.find("HMCTrj"), original.find("MCControl")
    if trj is None or mc is None:
        raise ValueError("missing HMCTrj/MCControl")
    only(trj, {"nrow", "Monomials", "Hamiltonian", "MDIntegrator"})
    md = trj.find("MDIntegrator")
    if md is None:
        raise ValueError("missing MDIntegrator")
    only(md, {"tau0", "copyList", "Integrator"})
    outer = md.find("Integrator")
    inner = None if outer is None else outer.find("SubIntegrator")
    if inner is None or inner.find("SubIntegrator") is not None:
        raise ValueError("expected exactly two integrator levels")
    for level in (outer, inner):
        only(level, {"Name", "n_steps", "monomial_ids", "SubIntegrator"})
        if required(level, "Name") != "LCM_STS_FORCE_GRAD":
            raise ValueError("only LCM_STS_FORCE_GRAD supported")
    def ids(node):
        result = [required(e, ".") for e in node.findall("monomial_ids/elem")]
        if not result or len(set(result)) != len(result):
            raise ValueError("empty/duplicate monomial_ids")
        return result
    ham = trj.find("Hamiltonian")
    if ham is None:
        raise ValueError("missing Hamiltonian")
    only(ham, {"monomial_ids"})
    ham_ids, outer_ids, inner_ids = ids(ham), ids(outer), ids(inner)
    if set(outer_ids) & set(inner_ids):
        raise ValueError("monomial occurs in both integrator levels")
    rows, elements = [], {}
    shared_state = None
    for m in trj.findall("Monomials/elem"):
        mid, name = required(m, "NamedObject/monomial_id"), required(m, "Name")
        if mid in elements:
            raise ValueError("duplicate monomial id: " + mid)
        if name not in ROLES:
            raise ValueError("unsupported monomial: " + name)
        role = ROLES[name]
        allowed = {"Name", "NamedObject", "ChronologicalPredictor"}
        allowed |= {"logdet": {"FermionAction", "num_flavors"},
                    "strange": {"Action", "num_pf"},
                    "hasenbusch_ratio": {"Action", "PrecAction"},
                    "hasenbusch_cancel": {"FermionAction", "InvertParam", "invParam"},
                    "gauge": {"GaugeAction"}}[role]
        only(m, allowed)
        only(m.find("NamedObject"), {"monomial_id"})
        if role == "hasenbusch_ratio":
            ferms = [m.find("Action/FermionAction"), m.find("PrecAction/FermionAction")]
        elif role == "strange":
            ferms = [m.find("Action/FermionAction")]
        else:
            ferms = [] if role == "gauge" else [m.find("FermionAction")]
        if any(f is None for f in ferms):
            raise ValueError("missing scalar Action/PrecAction/FermionAction: " + mid)
        for wrapper in (m.find("Action"), m.find("PrecAction")):
            if wrapper is not None:
                allowed_action = {"FermionAction", "ActionApprox", "ForceApprox"} if role == "strange" else {
                    "FermionAction", "InvertParam", "invParam", "ChronologicalPredictor"}
                only(wrapper, allowed_action)
                for approx_tag in ("ActionApprox", "ForceApprox"):
                    approx = wrapper.find(approx_tag)
                    if approx is not None:
                        only(approx, {"RationalApprox", "InvertParam", "invParam"})
        if len(ferms) != (2 if role == "hasenbusch_ratio" else 0 if role == "gauge" else 1):
            raise ValueError("unexpected FermionAction count: " + mid)
        info = [fermion_info(f) for f in ferms]
        for f in info:
            state = (f["stout"], f["boundary"])
            if shared_state is not None and state != shared_state:
                raise ValueError("different stout/BC require runner extension")
            shared_state = state
        if role == "hasenbusch_ratio" and (info[0]["mass"] >= info[1]["mass"] or info[0]["clover_csw"] != info[1]["clover_csw"]):
            raise ValueError("ratio requires increasing scalar masses and equal Clover coefficient")
        row = dict(id=mid, name=name, role=role, mass=info[0]["mass"] if info else None,
                   mass_precond=info[1]["mass"] if len(info) == 2 else None,
                   rational=rational(m) if role == "strange" else None,
                   placement="outer" if mid in outer_ids else "inner" if mid in inner_ids else "hamiltonian_only",
                   physical_xml=xml(m))
        if info:
            row.update(clover_csw=info[0]["clover_csw"], boundary=info[0]["boundary"],
                       fermion_actions=[xml(f) for f in ferms])
        if role == "logdet":
            row["n_flavor"] = integer(m, "num_flavors", 1)
            if row["n_flavor"] not in (1, 2):
                raise ValueError("only one/two flavor logdet supported")
        if role == "strange" and integer(m, "num_pf", 1) != 1:
            raise ValueError("only one rational pseudofermion supported")
        blocks = solver_info(m)
        if role in ("hasenbusch_ratio", "hasenbusch_cancel", "strange"):
            expected_blocks = 1 if role == "hasenbusch_cancel" else 2
            if len(blocks) != expected_blocks or len({(b["tol"], b["maxiter"]) for b in blocks}) != 1:
                raise ValueError("missing/different internal solver controls require runner extension: " + mid)
            if role != "strange" and any(len(b["tolerance"]) != 1 for b in blocks):
                raise ValueError("scalar solver requires one residual target")
            if role == "strange":
                for tag, sector in (("ActionApprox", "action"), ("ForceApprox", "force")):
                    sector_blocks = solver_info(m.find("Action/" + tag))
                    if len(sector_blocks) != 1 or len(sector_blocks[0]["tolerance"]) not in (1, len(row["rational"]["offset_" + sector])):
                        raise ValueError("rational per-pole tolerance count mismatch")
            row.update(tol=blocks[0]["tol"], maxiter=blocks[0]["maxiter"], solver_blocks=blocks)
        elif blocks:
            raise ValueError("unexpected solver in " + mid)
        rows.append(row)
        elements[mid] = m
    by_id = {r["id"]: r for r in rows}
    if set(ham_ids + outer_ids + inner_ids) - set(by_id):
        raise ValueError("unresolved monomial reference")
    copies = []
    canonical = {mid: mid for mid in by_id}
    for pair in md.findall("copyList/elem"):
        only(pair, {"copyFrom", "copyTo"})
        source, dest = required(pair, "copyFrom"), required(pair, "copyTo")
        if (source not in ham_ids or dest not in outer_ids + inner_ids or dest in ham_ids
                or canonical[dest] != dest or source == dest):
            raise ValueError("copyList must uniquely map Hamiltonian -> MD")
        a, b = copy.deepcopy(elements[source]), copy.deepcopy(elements[dest])
        for e in (a, b):
            e.remove(e.find("NamedObject"))
        if signature(a) != signature(b):
            raise ValueError("copyList physics mismatch: " + source + " -> " + dest)
        if by_id[source].get("maxiter") != by_id[dest].get("maxiter") and cancel_maxiter is None:
            raise ValueError("different copy iteration caps unsupported")
        canonical[dest] = source
        copies.append(dict(copyFrom=source, copyTo=dest))
    placed = [canonical[mid] for mid in outer_ids + inner_ids]
    if len(set(placed)) != len(placed) or set(placed) != set(ham_ids) or set(by_id) != set(ham_ids + outer_ids + inner_ids):
        raise ValueError("each Hamiltonian object needs exactly one MD placement via explicit copyList")
    seed_node = mc.find("RNG/Seed")
    seed = [int(x) for x in required(seed_node, ".").split()] if seed_node is not None and not list(seed_node) else [int(required(e, ".")) for e in mc.findall("RNG/Seed/elem")]
    if len(seed) != 4 or any(not 0 <= x < 4096 for x in seed) or seed[3] >= 2048:
        raise ValueError("expected four little-endian QDP 47-bit seed words")
    lattice = [int(x) for x in required(trj, "nrow").split()]
    if len(lattice) != 4 or any(x <= 0 or x % 2 for x in lattice):
        raise ValueError("expected four positive even lattice dimensions")
    gauges = [r for r in rows if r["role"] == "gauge" and r["id"] in ham_ids]
    strange = [r for r in rows if r["role"] == "strange" and r["id"] in ham_ids]
    ratios = [r for r in rows if r["role"] == "hasenbusch_ratio" and r["id"] in ham_ids]
    cancels = [r for r in rows if r["role"] == "hasenbusch_cancel" and r["id"] in ham_ids]
    if len(gauges) != 1 or len(strange) != 1 or not ratios or len(cancels) != 1:
        raise ValueError("need one LW gauge, one rational strange, scalar ratios and one cancel")
    chain = sorted(ratios, key=lambda r: r["mass"])
    if any(a["mass_precond"] != b["mass"] for a, b in zip(chain, chain[1:])) or chain[-1]["mass_precond"] != cancels[0]["mass"]:
        raise ValueError("scalar mass chain is not contiguous through cancel")
    light = chain[0]
    if any(r["clover_csw"] != light["clover_csw"] for r in ratios + cancels):
        raise ValueError("inconsistent light Clover coefficients")
    logdets = [r for r in rows if r["role"] == "logdet" and r["id"] in ham_ids]
    if sorted((r["n_flavor"], r["mass"], r["clover_csw"]) for r in logdets) != sorted([
            (2, light["mass"], light["clover_csw"]), (1, strange[0]["mass"], strange[0]["clover_csw"])]):
        raise ValueError("need matching light two-flavor and strange one-flavor logdet")
    gauge_action = elements[gauges[0]["id"]].find("GaugeAction")
    only(gauge_action, {"Name", "beta", "u0", "GaugeBC"})
    if required(gauge_action, "Name") != "LW_TREE_GAUGEACT" or required(gauge_action, "GaugeBC/Name") != "PERIODIC_GAUGEBC":
        raise ValueError("only periodic LW_TREE_GAUGEACT supported")
    only(gauge_action.find("GaugeBC"), {"Name"})
    cfg = mc.find("Cfg")
    if required(cfg, "cfg_type") != "SZINQIO":
        raise ValueError("only SZINQIO gauge input supported")
    reunit = boolean(cfg, "reunit") if cfg.find("reunit") is not None else False
    if not reunit:
        raise ValueError("current runner requires explicit Cfg/reunit=true")
    config = dict(schema_version=1, gauge_path=str(gauge), latt_size=lattice,
                  boundary=shared_state[1], t_boundary=shared_state[1][3], reunit=reunit,
                  cfg_xml=xml(cfg), beta=number(gauge_action, "beta"), u0=number(gauge_action, "u0"),
                  gauge_action="LW_TREE_GAUGEACT", gauge_bc="PERIODIC_GAUGEBC",
                  masses=[r["mass"] for r in chain] + [cancels[0]["mass"]],
                  clover_csw_light=light["clover_csw"], clover_csw_strange=strange[0]["clover_csw"],
                  stout=shared_state[0], seed_words=seed,
                  seed_packed=sum(x << (12 * i) for i, x in enumerate(seed)),
                  hamiltonian_order=ham_ids, copy_list=copies, monomials=rows,
                  integrator=dict(tau0=number(md, "tau0"), outer=integer(outer, "n_steps", 1),
                                  inner=integer(inner, "n_steps", 1),
                                  outer_type=required(outer, "Name"), inner_type=required(inner, "Name"),
                                  outer_monomials=outer_ids, inner_monomials=inner_ids),
                  sampling_order=[f"momentum[{i}]" for i in range(4)] + [mid for mid in ham_ids if by_id[mid]["role"] in ("strange", "hasenbusch_ratio", "hasenbusch_cancel")],
                  physical_header_xml=xml(original), ntraj=ntraj)
    if config["integrator"]["tau0"] <= 0 or config["beta"] <= 0 or not 0 < config["u0"] <= 1:
        raise ValueError("invalid tau/beta/u0")
    root = copy.deepcopy(original)
    changes, overrides = [], []
    for m in root.findall("HMCTrj/Monomials/elem"):
        mid = required(m, "NamedObject/monomial_id")
        for inv in m.iter():
            if inv.tag.lower() not in ("invparam", "invertparam"):
                continue
            old = xml(inv)
            target = inv.find("RsdTarget")
            if target is None:
                target = inv.find("RsdCG")
            cap = int(inv.findtext("MaxCG") or inv.findtext("MaxIter"))
            if cancel_maxiter is not None and by_id[mid]["role"] == "hasenbusch_cancel":
                overrides.append(dict(id=mid, old_maxiter=cap, maxiter=cancel_maxiter))
                cap = cancel_maxiter
                by_id[mid]["maxiter"] = cap
            set_text(inv, "invType", "CG_INVERTER")
            set_text(inv, "RsdCG", " ".join(target.itertext()).strip())
            set_text(inv, "MaxCG", cap)
            if inv.find("MaxIter") is not None:
                set_text(inv, "MaxIter", cap)
            changes.append(dict(id=mid, before=old, after=xml(inv)))
        for p in m.iter():
            if "predictor" in p.tag.lower():
                p.clear()
                ET.SubElement(p, "Name").text = "ZERO_GUESS_4D_PREDICTOR"
    # 求解器历史原文留在 solver_blocks；有效上限由 row.maxiter 明确给出。
    for pair in copies:
        if by_id[pair["copyFrom"]].get("maxiter") != by_id[pair["copyTo"]].get("maxiter"):
            raise ValueError("copy iteration caps still differ after override")
    out_mc = root.find("MCControl")
    set_text(out_mc.find("Cfg"), "cfg_file", str(gauge))
    start, warm = integer(mc, "StartUpdateNum"), integer(mc, "NWarmUpUpdates")
    if start < warm:
        raise ValueError("production-only tool cannot start during warmup")
    for key, value in dict(NProductionUpdates=start + ntraj - warm, NUpdatesThisRun=ntraj,
                           SaveInterval=1, SavePrefix=str(Path(output_dir) / "chroma")).items():
        set_text(out_mc, key, value)
    measurements = out_mc.find("InlineMeasurements")
    removed = xml(measurements) if measurements is not None else None
    if measurements is not None:
        measurements.clear()
    if signature(trj) != signature(root.find("HMCTrj")) or xml(mc.find("RNG")) != xml(out_mc.find("RNG")):
        raise ValueError("unexpected physics/integrator/RNG change")
    config.update(start_update=start, final_update=start + ntraj)
    audit = dict(schema_version=1, header_sha256=hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest(),
                 hamiltonian_order=ham_ids, copy_list=copies, inverters=changes,
                 cancel_maxiter_overrides=overrides, removed_inline_measurements=removed,
                 start_update=start, final_update=start + ntraj, ntraj=ntraj,
                 normalization="raw Clover coefficient and Chroma tau0 retained; no twist or time rescaling",
                 solver_policy="CG_INVERTER; equal per-pole targets retained verbatim; zero initial guess")
    return root, config, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauge", required=True)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ntraj", type=int, default=2)
    parser.add_argument("--cancel-maxiter", type=int)
    args = parser.parse_args(argv)
    raw = read_header(args.header if args.header is not None else args.gauge)
    root, config, audit = convert(raw, args.gauge, args.output_dir, args.cancel_maxiter, args.ntraj)
    output = args.output_dir
    # 防止 --header 或 --gauge 恰好指向将被覆盖的产物。
    inputs = {Path(args.gauge).resolve()}
    if args.header is not None:
        inputs.add(args.header.resolve())
    if any((output / name).resolve() in inputs for name in ("header.xml", "input.xml", "config.json", "audit.json")):
        raise ValueError("output would overwrite an input")
    output.mkdir(parents=True, exist_ok=True)
    (output / "header.xml").write_bytes(raw)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output / "input.xml", encoding="utf-8", xml_declaration=True)
    for name, obj in (("config.json", config), ("audit.json", audit)):
        (output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, ET.ParseError) as exc:
        sys.exit("extract_chroma_header: " + str(exc))
