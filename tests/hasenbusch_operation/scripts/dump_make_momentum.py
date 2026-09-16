#!/usr/bin/env python3
"""输出 ChromaRNG.make_momentum 的 LatticeMom 场，用于 B0 逐坐标对拍。"""

import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(OP_DIR.parent.parent))

from pyquda.chroma_rng import ChromaRNG
from pyquda_utils import core


LATT_SIZE = (12, 12, 12, 32)
SEED = 20260910
OUT = OP_DIR / "rng_results/rngB_pyquda_make_momentum_fixed_12x32.txt"


def coordinate_map(latt_info):
    coords = latt_info.coordinate()
    coords = coords.get() if hasattr(coords, "get") else coords
    lx, ly, lz, lt = latt_info.size
    lxh = lx // 2
    out = {}
    for slot in range(2):
        for t in range(lt):
            for z in range(lz):
                for y in range(ly):
                    for xh in range(lxh):
                        coord = tuple(int(v) for v in coords[:, slot, t, z, y, xh])
                        index = ((t * lz + z) * ly + y) * lxh + xh
                        out[coord] = (slot, index)
    return out


def ten_to_matrix(values):
    m = np.zeros((3, 3), dtype=np.complex128)
    m[0, 1] = values[0] + 1j * values[1]
    m[0, 2] = values[2] + 1j * values[3]
    m[1, 2] = values[4] + 1j * values[5]
    m[1, 0] = -np.conj(m[0, 1])
    m[2, 0] = -np.conj(m[0, 2])
    m[2, 1] = -np.conj(m[1, 2])
    m[0, 0] = 1j * values[6]
    m[1, 1] = 1j * values[7]
    m[2, 2] = 1j * values[8]
    return m


def main():
    cache = OP_DIR / ".cache" / "make_momentum_dump"
    cache.mkdir(parents=True, exist_ok=True)
    core.init(None, list(LATT_SIZE), resource_path=str(cache), enable_force_monitor=False)
    latt_info = core.LatticeInfo(list(LATT_SIZE), t_boundary=-1, anisotropy=1.0)
    rng = ChromaRNG(LATT_SIZE, precision="double")
    mom = rng.make_momentum(latt_info)
    print("DEBUG_AFTER_MAKE", mom.data[0, 0].reshape(-1, 10)[0].get())
    mapping = coordinate_map(latt_info)
    host = mom.data.get()

    with OUT.open("w") as f:
        header = {
            "format": "chroma_rng_dump_v1",
            "latt": list(LATT_SIZE),
            "seed": SEED,
            "mode": "mom",
            "precision": "double",
            "source": "pyquda_make_momentum_fixed",
        }
        f.write("# " + json.dumps(header) + "\n")
        first_values = host[0, mapping[(0, 0, 0, 0)][0]].reshape(-1, 10)[mapping[(0, 0, 0, 0)][1]]
        print("DEBUG_FIRST_VALUES", first_values)
        for coord in sorted(mapping):
            slot, index = mapping[coord]
            f.write(" ".join(str(v) for v in coord))
            for mu in range(4):
                values = host[mu, slot].reshape(-1, 10)[index]
                matrix = ten_to_matrix(values)
                for row in range(3):
                    for col in range(3):
                        f.write(f" {matrix[row, col].real:.17g} {matrix[row, col].imag:.17g}")
            f.write("\n")
    print(OUT)


if __name__ == "__main__":
    main()
