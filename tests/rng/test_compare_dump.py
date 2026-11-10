"""CPU unittest：在临时目录构造 v1 dump，覆盖完整性、元数据和位级/数值判定。

运行：python3 -B -m unittest discover -s tests/rng -p test_compare_dump.py -v。
不导入 PyQUDA/Chroma，不需要 MPI/GPU；临时输入自动清理。
"""

import contextlib
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


# 按文件加载，避免 unittest 从不同工作目录发现测试时依赖 sys.path。
SCRIPT = Path(__file__).with_name("compare_dump.py")
SPEC = importlib.util.spec_from_file_location("rng_compare_dump", SCRIPT)
compare_dump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_dump)


class CompareDumpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.left = Path(self.tmp.name) / "left.txt"
        self.right = Path(self.tmp.name) / "right.txt"

    def fixture(self, mode="uniforms", precision="double", latt=(2, 2, 2, 2)):
        header = dict(format="chroma_rng_dump_v1", latt=list(latt), seed=20260910,
                      mode=mode, precision=precision)
        rows = []
        for coord in itertools.product(*(range(n) for n in latt)):
            if mode == "mom" or sum(coord) % 2:
                rows.append((coord, [0.25] * compare_dump.COMPONENTS[mode]))
        return header, rows

    def write(self, path, header, rows):
        lines = ["# " + json.dumps(header)]
        lines.extend(" ".join(map(str, (*coord, *values))) for coord, values in rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def pair(self, mode="uniforms", precision="double"):
        header, rows = self.fixture(mode, precision)
        self.write(self.left, header, rows)
        self.write(self.right, header, rows)
        return header, rows

    def test_identical_all_modes(self):
        for mode in compare_dump.COMPONENTS:
            with self.subTest(mode=mode):
                self.pair(mode)
                self.assertIn(f"mode={mode}", compare_dump.compare_dumps(self.left, self.right))

    def test_order_and_source_are_irrelevant(self):
        header, rows = self.pair()
        header["source"] = "chroma"
        self.write(self.right, header, list(reversed(rows)))
        compare_dump.compare_dumps(self.left, self.right)

    def test_duplicate_coordinate_rejected(self):
        header, rows = self.pair()
        for replacement in (rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.subTest(length=len(replacement)):
                self.write(self.right, header, replacement)
                with self.assertRaisesRegex(ValueError, "duplicate coordinate"):
                    compare_dump.compare_dumps(self.left, self.right)

    def test_missing_coordinates_even_on_both_sides_rejected(self):
        for mode in compare_dump.COMPONENTS:
            for both in (False, True):
                with self.subTest(mode=mode, both=both):
                    header, rows = self.pair(mode)
                    self.write(self.right, header, rows[:-1])
                    if both:
                        self.write(self.left, header, rows[:-1])
                    with self.assertRaisesRegex(ValueError, "incomplete coordinate set"):
                        compare_dump.compare_dumps(self.left, self.right)

    def test_component_count_rejected(self):
        for mode in compare_dump.COMPONENTS:
            for delta in (-1, 1):
                with self.subTest(mode=mode, delta=delta):
                    header, rows = self.pair(mode)
                    rows[0] = (rows[0][0], [0.25] * (compare_dump.COMPONENTS[mode] + delta))
                    self.write(self.right, header, rows)
                    with self.assertRaisesRegex(ValueError, "components"):
                        compare_dump.compare_dumps(self.left, self.right)

    def test_nonfinite_components_rejected(self):
        for mode in compare_dump.COMPONENTS:
            for value in (math.nan, math.inf, -math.inf, "1e999"):
                with self.subTest(mode=mode, value=value):
                    header, rows = self.pair(mode)
                    rows[0][1][0] = value
                    self.write(self.right, header, rows)
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        compare_dump.compare_dumps(self.left, self.right)

    def test_uniform_float32_overflow_rejected(self):
        header, rows = self.pair()
        rows[0][1][0] = 1e100
        self.write(self.right, header, rows)
        with self.assertRaisesRegex(ValueError, "nonfinite float32"):
            compare_dump.compare_dumps(self.left, self.right)

    def test_uniform_one_bit_difference_rejected(self):
        header, rows = self.pair()
        bits = struct.unpack("!I", struct.pack("!f", 0.25))[0]
        rows[0][1][0] = struct.unpack("!f", struct.pack("!I", bits + 1))[0]
        self.write(self.right, header, rows)
        with self.assertRaisesRegex(ValueError, "bit mismatch"):
            compare_dump.compare_dumps(self.left, self.right)

    def test_uniform_signed_zero_rejected(self):
        header, rows = self.pair()
        rows[0][1][0] = 0.0
        self.write(self.left, header, rows)
        rows[0][1][0] = -0.0
        self.write(self.right, header, rows)
        with self.assertRaisesRegex(ValueError, "bit mismatch"):
            compare_dump.compare_dumps(self.left, self.right)

    def test_uniform_rounding_uses_float32(self):
        header, rows = self.pair()
        rows[0][1][0] += 1e-10
        self.write(self.right, header, rows)
        compare_dump.compare_dumps(self.left, self.right)

    def test_gaussian_double_scaled_tolerance(self):
        for mode in ("eta", "mom"):
            for value in (0.0, -0.25, 100.0):
                for error, passed in ((0.5e-12, True), (2e-12, False)):
                    with self.subTest(mode=mode, value=value, error=error):
                        header, rows = self.pair(mode)
                        rows[0][1][0] = value
                        self.write(self.left, header, rows)
                        rows[0][1][0] += error * max(1.0, abs(value))
                        self.write(self.right, header, rows)
                        if passed:
                            compare_dump.compare_dumps(self.left, self.right)
                        else:
                            with self.assertRaisesRegex(ValueError, "tolerance 1e-12 exceeded"):
                                compare_dump.compare_dumps(self.left, self.right)

    def test_gaussian_negative_zero_is_numerical(self):
        header, rows = self.pair("eta")
        rows[0][1][0] = 0.0
        self.write(self.left, header, rows)
        rows[0][1][0] = -0.0
        self.write(self.right, header, rows)
        compare_dump.compare_dumps(self.left, self.right)

    def test_single_precision_historical_tolerance(self):
        for error, passed in ((1e-6, True), (3e-6, False)):
            with self.subTest(error=error):
                header, rows = self.pair("eta", "single")
                rows[0][1][0] += error
                self.write(self.right, header, rows)
                if passed:
                    compare_dump.compare_dumps(self.left, self.right)
                else:
                    with self.assertRaisesRegex(ValueError, "tolerance 2e-06 exceeded"):
                        compare_dump.compare_dumps(self.left, self.right)

    def test_metadata_mismatch_rejected(self):
        for key, value in (("mode", "eta"), ("seed", 123), ("precision", "single"),
                           ("latt", [2, 2, 2, 4])):
            with self.subTest(key=key):
                header, rows = self.pair()
                header[key] = value
                if key == "latt":
                    _, rows = self.fixture(latt=value)
                self.write(self.right, header, rows)
                with self.assertRaisesRegex(ValueError, f"{key} mismatch"):
                    compare_dump.compare_dumps(self.left, self.right)

    def test_unknown_or_missing_precision_keeps_double_tolerance(self):
        for precision in (None, "unknown"):
            with self.subTest(precision=precision):
                header, rows = self.pair("eta")
                if precision is None:
                    header.pop("precision")
                else:
                    header["precision"] = precision
                self.write(self.right, header, rows)
                compare_dump.compare_dumps(self.left, self.right)
                rows[0][1][0] += 1e-8
                self.write(self.right, header, rows)
                with self.assertRaisesRegex(ValueError, "tolerance 1e-12 exceeded"):
                    compare_dump.compare_dumps(self.left, self.right)

    def test_invalid_header_rejected(self):
        changes = [("format", "v2"), ("mode", "bad"), ("mode", []),
                   ("latt", [2, 2, 2]), ("latt", [0, 2, 2, 2]),
                   ("latt", [True, 2, 2, 2]), ("latt", [2.0, 2, 2, 2]),
                   ("seed", -1), ("seed", 2**47), ("seed", True),
                   ("seed", "11"), ("precision", "half")]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                header, rows = self.pair()
                header[key] = value
                self.write(self.right, header, rows)
                with self.assertRaises(ValueError):
                    compare_dump.load_dump(self.right)

    def test_missing_required_metadata_rejected(self):
        for key in ("format", "mode", "latt", "seed"):
            with self.subTest(key=key):
                header, rows = self.pair()
                del header[key]
                self.write(self.right, header, rows)
                with self.assertRaises(ValueError):
                    compare_dump.load_dump(self.right)

    def test_bad_coordinates_rejected(self):
        for coord in ((-1, 0, 0, 0), (2, 1, 0, 0), (0, 0, 0, 0), ("x", 0, 0, 0)):
            with self.subTest(coord=coord):
                header, rows = self.pair()
                rows[0] = (coord, rows[0][1])
                self.write(self.right, header, rows)
                with self.assertRaises(ValueError):
                    compare_dump.load_dump(self.right)

    def test_empty_and_malformed_dump_rejected(self):
        for content in ("", "\n", "# {}\n", "# []\n", "# {\n", "0 0 0 1 0.25\n",
                        '# {"seed":1,"seed":2}\n', '# {"seed":NaN}\n'):
            with self.subTest(content=content):
                self.right.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    compare_dump.load_dump(self.right)

    def test_header_only_rejected(self):
        header, _ = self.fixture()
        self.write(self.right, header, [])
        with self.assertRaisesRegex(ValueError, "incomplete coordinate set"):
            compare_dump.load_dump(self.right)

    def test_repeated_rank_headers_and_whitespace(self):
        header, rows = self.pair()
        self.write(self.right, header, rows[:4])
        first = self.right.read_text(encoding="utf-8")
        self.write(self.right, header, rows[4:])
        second = self.right.read_text(encoding="utf-8")
        self.right.write_text("\n  " + first + "\n" + second, encoding="utf-8")
        compare_dump.compare_dumps(self.left, self.right)
        header["seed"] += 1
        self.right.write_text(first + "# " + json.dumps(header) + "\n" + second, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inconsistent repeated header"):
            compare_dump.load_dump(self.right)

    def test_optional_metadata_consistency(self):
        header, rows = self.pair()
        header["shape"] = [8, 24]
        self.write(self.left, header, rows)
        header["shape"] = [8, 72]
        self.write(self.right, header, rows)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compare_dump.compare_dumps(self.left, self.right)

    def test_odd_lattice_subset_count(self):
        header, rows = self.fixture(latt=(3, 3, 3, 3))
        self.write(self.left, header, rows)
        self.assertEqual(len(compare_dump.load_dump(self.left)[1]), 40)

    def test_main_exit_codes(self):
        self.pair()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare_dump.main([str(self.left), str(self.right)]), 0)
            self.assertEqual(compare_dump.main([]), 2)
            self.assertEqual(compare_dump.main([str(self.left), "missing-file"]), 1)

    def test_optimized_python_still_rejects_bad_dump(self):
        header, rows = self.pair()
        self.write(self.right, header, rows[:-1])
        result = subprocess.run([sys.executable, "-B", "-O", str(SCRIPT), str(self.left), str(self.right)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("incomplete coordinate set", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class XMLTemplateTests(unittest.TestCase):
    def test_preserved_parameters_and_portable_paths(self):
        path = Path(__file__).with_name("chroma_12x32.xml")
        root = ET.parse(path).getroot()
        self.assertEqual(root.findtext("HMCTrj/nrow"), "12 12 12 32")
        self.assertEqual(root.findtext("HMCTrj/MDIntegrator/tau0"), "0.1")
        self.assertEqual([e.text for e in root.findall("MCControl/RNG/Seed/elem")],
                         ["2094", "850", "1", "0"])
        for xpath in ("MCControl/Cfg/cfg_file", "MCControl/SavePrefix"):
            self.assertFalse(Path(root.findtext(xpath)).is_absolute())
        comments = path.read_text(encoding="utf-8")
        self.assertIn("25d5e347", comments)
        self.assertIn("20260916", comments)
        self.assertIn("t=tau0/sqrt(2)", comments)
        for coeff in root.findall(".//PFECoeffs") + root.findall(".//IPFECoeffs"):
            self.assertEqual(len(coeff.findtext("res").split()), len(coeff.findtext("pole").split()))


if __name__ == "__main__":
    unittest.main()
