"""CG/MG 求解管理：输入 action 参数与半场 RHS，输出同一正规方程的解。

MG 默认关闭；仅 MD 可使用 MG，sample/action 使用 CG。MG 总是复核原始伴随与
正规方程残差；FG 只借用已有基，不刷新、不重建、不回退。调用 close() 显式释放池。
种子默认要求已加载库的 seed API；builtin_fixed1234 仅接受显式 path/hash 证明。
证据记录默认关闭，显式开启后仅记录求解/MG 生命周期，可写独立 JSON；不使用 predictor，
不修改物理 HMC RNG，不扫描实验目录或保存计时历史。
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import count
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
from weakref import WeakSet

from .enum_quda import (
    QUDA_MAX_DIM, QUDA_MAX_MG_LEVEL, QudaBoolean, QudaDagType, QudaDslashType,
    QudaGammaBasis, QudaInverterType, QudaMatPCType, QudaParity,
    QudaPrecision, QudaSchwarzType, QudaSolutionType,
    QudaSolveType, QudaTwistFlavorType, QudaUseInitGuess,
)


from .field import LatticeFermion
from .quda import MatQuda, cloverQuda, invertQuda

_VALIDATION_ENABLED = False
_TELEMETRY_ENABLED = False
_TELEMETRY_PATH = None
_TRAJECTORY = 0
_TELEMETRY = {"schema_version": 1, "events": [], "counts": {}}


def configure_solver_accel_telemetry(path=None, *, enabled=True):
    """重置本进程证据；path 可省略，只有 write 显式调用时 rank 0 才写文件。"""
    global _TELEMETRY_ENABLED, _TELEMETRY_PATH, _TRAJECTORY
    if type(enabled) is not bool:
        raise TypeError("telemetry.enabled requires a boolean")
    _TELEMETRY_ENABLED, _TELEMETRY_PATH, _TRAJECTORY = enabled, path, 0
    _TELEMETRY.update(events=[], counts={})


def set_solver_accel_trajectory(trajectory=None):
    """仅标注证据所属轨迹，不消耗 RNG。"""
    global _TRAJECTORY
    _TRAJECTORY = _TRAJECTORY + 1 if trajectory is None else int(trajectory)


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    return value


def _emit_telemetry(event, **values):
    if _TELEMETRY_ENABLED:
        _TELEMETRY["events"].append(_json_finite(dict(event=event, trajectory=_TRAJECTORY, **values)))
        _TELEMETRY["counts"][event] = _TELEMETRY["counts"].get(event, 0) + 1


def get_solver_accel_telemetry():
    """返回独立副本；真实残差的 NaN/Inf 记 null，不能被门禁当作成功。"""
    return json.loads(json.dumps(_TELEMETRY, allow_nan=False))


def write_solver_accel_telemetry(path=None):
    """仅 rank 0 导出当前进程证据；调用方负责各 rank 的门禁和错误汇总。"""
    target = path or _TELEMETRY_PATH
    if not _TELEMETRY_ENABLED or target is None:
        return
    from pyquda_comm import getMPIComm

    if getMPIComm().Get_rank() == 0:
        output = Path(target).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(_TELEMETRY, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def set_solver_accel_validation(enabled: bool):
    """进程级显式门禁，默认关闭；所有 MPI rank 应在求解前设置同值，不改变精度或加速配置。"""
    global _VALIDATION_ENABLED
    if type(enabled) is not bool:
        raise TypeError("solver acceleration validation requires a boolean")
    _VALIDATION_ENABLED = enabled


class SolverValidationError(RuntimeError):
    """未通过真实正规残差门禁；detail 保留当前 shift/tol/scope，供 campaign 记录失败。"""

    def __init__(self, detail):
        self.detail = dict(detail)
        super().__init__(f"Solver normal residual validation failed: {self.detail}")


@dataclass(frozen=True)
class PredictorConfig:
    """仅兼容旧配置中关闭的 predictor；不保留历史槽与实验双阶段路径。"""

    enabled: bool = False
    max_history: int = 8
    precision: str = "double"
    formulation: str = "normal"

    def __post_init__(self):
        if self.enabled is not False or self.formulation != "normal":
            raise ValueError("predictor optimization is not supported; disable it and use formulation='normal'")
        if type(self.max_history) is not int or self.max_history < 1 or self.precision != "double":
            raise ValueError("invalid disabled predictor compatibility fields")

@dataclass(frozen=True)
class MultigridConfig:
    enabled: bool = False
    # 默认沿用 H20：FG 借用正常基；namespace 与 HMC 物理 stout context 分离。
    force_gradient: bool = True
    pool_namespace: str = None  # None 表示默认共享；保留 Python 3.9 可执行注解。
    block_size: tuple = (2, 2, 2, 2)
    n_vec: int = 24
    setup_maxiter: int = 500
    refresh_maxiter: int = 100
    setup_tol: float = 1e-6
    refresh_threshold: int = 100
    setup_seed: int = 20260910
    # api 必须找到真实 seed API；builtin 模式只证明指定库使用固定 1234 初始化。
    seed_backend: str = "api"
    library_path: str = None
    library_sha256: str = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("multigrid.enabled must be a boolean")
        if type(self.force_gradient) is not bool:
            raise ValueError("multigrid.force_gradient must be a boolean")
        if self.pool_namespace is not None and (not isinstance(self.pool_namespace, str) or not self.pool_namespace.strip()):
            raise ValueError("multigrid.pool_namespace must be a nonempty string or None")
        if len(self.block_size) != 4 or any(type(v) is not int or v < 1 for v in self.block_size):
            raise ValueError("multigrid.block_size must contain four positive integers")
        object.__setattr__(self, "block_size", tuple(self.block_size))
        for name in ("n_vec", "setup_maxiter", "refresh_maxiter", "refresh_threshold"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"multigrid.{name} must be a positive integer")
        if not math.isfinite(self.setup_tol) or not 0 < self.setup_tol < 1:
            raise ValueError("multigrid.setup_tol must be finite and between 0 and 1")
        if type(self.setup_seed) is not int or not 0 <= self.setup_seed < 2**64:
            raise ValueError("multigrid.setup_seed must be a uint64 integer")

        if self.seed_backend not in ("api", "builtin_fixed1234"):
            raise ValueError("multigrid.seed_backend must be api or builtin_fixed1234")
        if self.seed_backend == "builtin_fixed1234":
            if self.setup_seed != 1234:
                raise ValueError("builtin_fixed1234 requires setup_seed=1234")
            if not isinstance(self.library_path, str) or not os.path.isabs(self.library_path):
                raise ValueError("builtin_fixed1234 requires an absolute library_path")
            if (not isinstance(self.library_sha256, str) or len(self.library_sha256) != 64
                    or any(c not in "0123456789abcdef" for c in self.library_sha256)):
                raise ValueError("builtin_fixed1234 requires a lowercase SHA256 library_sha256")
        elif self.library_path is not None or self.library_sha256 is not None:
            raise ValueError("library_path/library_sha256 are only used by builtin_fixed1234")

@dataclass(frozen=True)
class SolverAccelConfig:
    """plain_precision 可在首次求解前经 configure 应用，求解后改变精度需重建 action。"""

    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    multigrid: MultigridConfig = field(default_factory=MultigridConfig)
    plain_precision: str = "default"

    def __post_init__(self):
        if self.plain_precision not in ("default", "double"):
            raise ValueError("plain_precision must be 'default' or 'double'")

    @classmethod
    def from_dict(cls, config=None):
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if not isinstance(config, dict) or set(config) - {"predictor", "multigrid", "plain_precision"}:
            raise ValueError("solver acceleration config requires predictor/multigrid objects and optional plain_precision")
        try:
            return cls(PredictorConfig(**config.get("predictor", {})),
                       MultigridConfig(**config.get("multigrid", {})), config.get("plain_precision", "default"))
        except TypeError as exc:
            raise ValueError(f"invalid solver acceleration fields: {exc}") from exc

    @classmethod
    def from_json(cls, path):
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as source:
            return cls.from_dict(json.load(source))

    def as_dict(self):
        config = asdict(self)
        config["multigrid"]["block_size"] = list(self.multigrid.block_size)
        # 旧配置继续只导出 predictor/multigrid 两键；只有显式 double 才加入新键。
        if self.plain_precision == "default":
            del config["plain_precision"]
        return config


class UnsupportedMultigridTopology(NotImplementedError):
    """局部几何会让 QUDA 缩小请求块时明确拒绝，禁止静默改变 MG 配置。"""


class _TemporaryMGConflict(RuntimeError):
    """同池 FG scope 尚未退出，普通调用不得刷新或重建共享 B。"""


def validate_multigrid_topology(latt_info, config):
    local_dims = tuple(int(value) for value in latt_info.size)
    block_size = tuple(config.block_size)
    # transfer.cpp 会在 coarse 局部维度为奇数时减半聚合块；block2 必须 local_dims % 4 == 0。
    incompatible = [dimension for dimension, (local, block) in enumerate(zip(local_dims, block_size))
                    if local <= 0 or local % (2 * block)]
    if len(local_dims) != 4 or incompatible:
        grid_size = tuple(int(value) for value in latt_info.grid_size)
        reason = (f"Unsupported MG topology: local_dims={local_dims}, grid_size={grid_size}, "
                  f"requested block_size={block_size}; each local dimension must be "
                  "divisible by twice its requested block; QUDA block reduction is forbidden")
        raise UnsupportedMultigridTopology(reason)
    return local_dims


@contextmanager
def parameter_scope(param, **values):
    """异常时也恢复共享 invert_param，避免 action 能量或 force 使用 MD 参数。"""
    saved = {name: getattr(param, name) for name in values}
    try:
        for name, value in values.items():
            setattr(param, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(param, name, value)


def _mg_adjoint_parity(param):
    """仅接受已验证的 symmetric Clover PC 相似关系，普通/asymmetric MG 显式拒绝。"""
    parities = {
        QudaMatPCType.QUDA_MATPC_EVEN_EVEN: QudaParity.QUDA_EVEN_PARITY,
        QudaMatPCType.QUDA_MATPC_ODD_ODD: QudaParity.QUDA_ODD_PARITY,
    }
    if param.matpc_type not in parities or param.solution_type != QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION:
        raise NotImplementedError("MG adjoint requires symmetric even/even or odd/odd normal PC equations")
    if param.dslash_type not in (QudaDslashType.QUDA_CLOVER_WILSON_DSLASH,
                                 QudaDslashType.QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH):
        raise NotImplementedError("MG adjoint similarity is only validated for base/Hasenbusch Clover")
    if param.gamma_basis != QudaGammaBasis.QUDA_DEGRAND_ROSSI_GAMMA_BASIS:
        raise NotImplementedError("MG adjoint gamma5 requires DeGrandRossi gamma basis")
    return parities[param.matpc_type]


def _mg_clover(out, inp, param, parity, inverse):
    """当前 half 场存储始终位于 even_ptr；parity 指定实际驻留 Clover 的棋盘。"""
    with parameter_scope(param, dslash_type=QudaDslashType.QUDA_CLOVER_WILSON_DSLASH,
                         dagger=QudaDagType.QUDA_DAG_NO):
        cloverQuda(out.even_ptr, inp.even_ptr, param, parity, int(inverse))


def _mg_gamma5(out, inp):
    """DeGrandRossi 的 Gamma(15) 作用于当前 half 场；输入输出必须不同。"""
    from .gamma import Gamma

    gamma5 = Gamma(15)
    for spin in range(4):
        out.even.data[..., spin, :] = gamma5.sparse_data[spin] * inp.even.data[..., gamma5.sparse_indices[spin], :]


# 进程内串行池；基础 Clover 签名相同才共享，shift 不改变基底身份。
_CHRONO_OFF = dict(chrono_use_resident=0, chrono_make_resident=0, chrono_replace_last=0,
                   use_init_guess=QudaUseInitGuess.QUDA_USE_INIT_GUESS_NO)
_MG_POOL = {}
_MG_IDS = count(1)
_MG_ACTIVE = None
_RESIDENT_EPOCH = 0
_BASE_FIELDS = (
    "mass", "kappa", "m5", "Ls", "matpc_type", "mass_normalization", "solver_normalization",
    "clover_coeff", "clover_csw", "clover_rho", "clover_order", "clover_location",
    "gamma_basis", "dirac_order", "input_location", "output_location", "verbosity",
)


_PRECISION_FIELDS = tuple(prefix + name for prefix in ("", "clover_") for name in (
    "cpu_prec", "cuda_prec", "cuda_prec_sloppy", "cuda_prec_precondition",
    "cuda_prec_refinement_sloppy", "cuda_prec_eigensolver",
))


def _immutable(value):
    """复制可变配置的值签名，避免原地修改 stout/geometry 后旧 key 也跟着变化。"""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return tuple(sorted((key, _immutable(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_immutable(item) for item in value)
    hash(value)
    return value


@dataclass
class _MGEntry:
    key: tuple
    entry_id: int = field(default_factory=lambda: next(_MG_IDS))
    users: WeakSet = field(default_factory=WeakSet)
    mg: object = None
    dirty: bool = True
    refresh_pending: bool = False
    deferred_rebuild: bool = False
    resident_epoch: int = -1


def _resident_changed():
    global _RESIDENT_EPOCH, _MG_ACTIVE
    # updateClover 可能重分配 resident 指针，即使物理 gauge 未变也不能跳过 full update。
    _RESIDENT_EPOCH += 1
    _MG_ACTIVE = None


def _mapped_quda_library_path():
    """严格检查当前 Linux 进程唯一 libquda 的路径、设备和 inode，拒绝删改后的映射。"""
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("cannot verify loaded libquda without /proc/self/maps") from exc
    mapped = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5]
        if not os.path.basename(path).startswith("libquda.so"):
            continue
        if path.endswith(" (deleted)"):
            raise RuntimeError("loaded libquda mapping was deleted; restart before MG")
        resolved = os.path.realpath(path)
        stat = os.stat(resolved)
        device = tuple(int(part, 16) for part in fields[3].split(":"))
        if stat.st_ino != int(fields[4]) or device != (os.major(stat.st_dev), os.minor(stat.st_dev)):
            raise RuntimeError("loaded libquda differs from the current on-disk file")
        mapped[resolved] = resolved
    if len(mapped) != 1:
        raise RuntimeError(f"MG requires exactly one already-loaded libquda; mapped={sorted(mapped)}")
    return next(iter(mapped))


def _set_mg_setup_seed(config):
    """每次 setup/update 重新验证实际映射；固定种子证明不会伪装为可重设 RNG 的 API。"""
    path = _mapped_quda_library_path()
    if config.seed_backend == "builtin_fixed1234":
        if path != os.path.realpath(config.library_path):
            raise RuntimeError("builtin_fixed1234 library_path does not match loaded libquda")
        # 不缓存 hash：不能让替换库或变更配置绕过实际库身份验证。
        digest = hashlib.sha256()
        with open(path, "rb") as library_file:
            for chunk in iter(lambda: library_file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != config.library_sha256:
            raise RuntimeError("builtin_fixed1234 library_sha256 does not match loaded libquda")
        return
    try:
        library = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_NOW)
        setter = library.hmc_quda_set_mg_setup_seed
    except (AttributeError, OSError) as exc:
        raise RuntimeError("MG requires hmc_quda_set_mg_setup_seed(uint64_t); "
                           "or explicit builtin_fixed1234 library attestation") from exc
    setter.argtypes = [ctypes.c_uint64]
    setter.restype = ctypes.c_int
    status = setter(config.setup_seed)
    if status != 0:
        raise RuntimeError(f"hmc_quda_set_mg_setup_seed failed: status={status}, seed={config.setup_seed}")


class SolverAcceleration:
    """基础 Clover MG 共享池；configure 仅在首次求解前允许改变普通 CG 精度。"""

    def __init__(self, action, config=None):
        self.latt_info = action.latt_info
        self._dirac = action.dirac
        self.label = getattr(action, "label", "unlabeled")
        self.scope = "md"
        self.operator_context = None
        self._config = SolverAccelConfig()
        self._used = False
        self._suspended = 0
        self._suspend_config = None
        self._entry = None
        self._unbound_dirty = True
        self._unbound_refresh = False
        self.last_solve = None
        self.last_validation = None
        self.configure(config)

    def configure(self, config=None):
        """HMC 在构造 actions 后、首次求解前调用；后续禁止改精度，避免 resident 不一致。"""
        config = SolverAccelConfig.from_dict(config)
        if self._suspended:
            raise RuntimeError("cannot configure solver during a temporary scope")
        changed_precision = config.plain_precision != self.config.plain_precision
        if changed_precision and self._used:
            raise ValueError("plain_precision cannot change after the first solve; rebuild the action")
        if changed_precision:
            precision = QudaPrecision.QUDA_DOUBLE_PRECISION
            options = (dict(cuda=precision, sloppy=precision, refinement_sloppy=precision,
                            precondition=precision, eigensolver=precision)
                       if config.plain_precision == "double" else {})
            self._dirac.setPrecision(False, **options)
            _resident_changed()
        self._config = config

    @property
    def config(self):
        return self._config


    @config.setter
    def config(self, value):
        config = SolverAccelConfig.from_dict(value)
        # 直接赋值只允许改策略；精度变化必须经 configure 同步更新 Dirac。
        if self._suspended:
            raise RuntimeError("cannot configure solver during a temporary scope")
        if config.plain_precision != self._config.plain_precision:
            raise ValueError("plain_precision changes require configure() before the first solve")
        self._config = config



    @property
    def _mg(self):
        return self._entry.mg if self._entry is not None else None


    @property
    def _mg_dirty(self):
        entry = self._entry
        return (self._unbound_dirty if entry is None else
                entry.dirty or entry.resident_epoch != _RESIDENT_EPOCH or _MG_ACTIVE is not entry)


    @_mg_dirty.setter
    def _mg_dirty(self, value):
        self._unbound_dirty = bool(value)
        if self._entry is not None:
            self._entry.dirty = bool(value)


    @property
    def _refresh_pending(self):
        return self._entry.refresh_pending if self._entry is not None else self._unbound_refresh


    @_refresh_pending.setter
    def _refresh_pending(self, value):
        if self._entry is None:
            self._unbound_refresh = bool(value)
        else:
            self._entry.refresh_pending = bool(value)


    def _pool_key(self, fine):
        geometry = tuple(_immutable(getattr(self.latt_info, name, None)) for name in (
            "size", "global_size", "latt_size", "lattice_size", "grid_size", "grid_coord",
            "Nd", "Ns", "Nc", "mpi_size", "t_boundary", "anisotropy",
        ))
        return (id(self.latt_info), geometry, _immutable(self.operator_context),
                tuple(getattr(fine, name) for name in _BASE_FIELDS + _PRECISION_FIELDS),
                self.config.multigrid)


    def _defer(self, reason, **values):
        if self._entry is not None:
            self._entry.dirty = True
            self._entry.deferred_rebuild = True
        self._pool_event("deferred_rebuild", reason=reason, **values)

    def _pool_event(self, event, **values):
        _emit_telemetry(event, action=self.label, scope="temporary" if self._suspended else self.scope,
                        pool_id=self._entry.entry_id if self._entry is not None else None, **values)


    def _require_normal_pool_available(self, entry):
        if not self._suspended and entry is not None and any(user._suspended for user in entry.users):
            raise _TemporaryMGConflict("normal MG solve requires all shared temporary scopes to exit")


    def invalidate(self, reason="operator change", *, operator_changed=False):
        """共享参数变化只置脏；临时 scope 内保留正常场基底并推迟销毁。"""
        self.last_solve = None
        self.last_validation = None
        if self._suspended:
            if operator_changed:
                self._defer(reason)
            self._mg_dirty = True
        elif operator_changed:
            self._destroy_mg()
        else:
            self._mg_dirty = True

    def close(self):
        if self._suspended:
            self._defer("close during temporary scope")
            raise RuntimeError("cannot close shared MG during a temporary scope")
        self._destroy_mg()
        self.last_solve = None
        self.last_validation = None

    @contextmanager
    def suspended(self):
        """FG 借用正常 B，只更新 fine/coarse；不刷新、重建。"""
        if not self._suspended:
            self._suspend_config = SolverAccelConfig.from_dict(self.config)
            self._mg_dirty = True
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1
            if not self._suspended:
                self._suspend_config = None
            self._mg_dirty = True


    def gauge_updated(self):
        _resident_changed()
        self._mg_dirty = True


    def _destroy_mg(self):
        global _MG_ACTIVE
        entry, self._entry = self._entry, None
        if entry is not None:
            entry.users.discard(self)
            if not entry.users:
                if entry.mg is not None:
                    entry.mg.destroy()
                if _MG_ACTIVE is entry:
                    _MG_ACTIVE = None
                _MG_POOL.pop(entry.key, None)
        self._unbound_dirty = True
        self._unbound_refresh = False


    def _bind_pool(self, fine):
        key = self._pool_key(fine)
        if self._suspended:
            # 冷启动或签名变化不能偷偷在 FG 场上建立正常 B。
            if self._entry is not None and self._entry.key != key:
                raise RuntimeError("temporary MG operator/config signature changed")
            entry = self._entry or _MG_POOL.get(key)
            if entry is None or entry.mg is None or entry.deferred_rebuild:
                raise RuntimeError("temporary MG requires an existing healthy normal-field basis")
        else:
            self._require_normal_pool_available(self._entry)
            self._require_normal_pool_available(_MG_POOL.get(key))
            if self._entry is not None and self._entry.key != key:
                self._destroy_mg()
            entry = _MG_POOL.get(key)
            if entry is None:
                entry = _MG_POOL[key] = _MGEntry(key)
        if self._entry is None:
            self._entry = entry
            entry.users.add(self)
            if not self._suspended:
                entry.refresh_pending |= self._unbound_refresh
            self._unbound_refresh = False
        return entry


    def _discard_pool_mg(self, reason):
        global _MG_ACTIVE
        self._require_normal_pool_available(self._entry)
        if self._suspended:
            raise _TemporaryMGConflict("temporary MG must not discard the normal basis")
        entry = self._entry
        if entry is not None:
            if entry.mg is not None:
                entry.mg.destroy()
            entry.mg = None
            entry.dirty = True
            entry.refresh_pending = False
            entry.deferred_rebuild = False
            if _MG_ACTIVE is entry:
                _MG_ACTIVE = None


    def _prepare_mg(self, fine, rebuild=False):
        global _MG_ACTIVE
        # 必须先检查实际 MPI 局部尺寸，再分配或重建 MG，不能把失败当作收敛重试。
        validate_multigrid_topology(self.latt_info, self.config.multigrid)
        from .dirac.general import Multigrid, newQudaMultigridParam

        entry = self._bind_pool(fine)
        if self._suspended and rebuild:
            raise RuntimeError("temporary MG must not rebuild the normal basis")
        if rebuild or entry.deferred_rebuild:
            self._pool_event("rebuild")
            self._discard_pool_mg("shared MG rebuild")
        if self._mg is None:
            cfg = self.config.multigrid
            mg_param, base = newQudaMultigridParam(
                QudaDslashType.QUDA_CLOVER_WILSON_DSLASH, fine.mass, fine.kappa, [list(cfg.block_size)]
            )
            # 只替换 dslash/mu；保持 fine 的 kappa、clover 归一化、PC parity 和场布局。
            for name in _BASE_FIELDS:
                setattr(base, name, getattr(fine, name))
            base.dslash_type = QudaDslashType.QUDA_CLOVER_WILSON_DSLASH
            base.mu = 0.0
            base.twist_flavor = QudaTwistFlavorType.QUDA_TWIST_NO
            # QUDA multigrid_solver 构造器仅接受 full DIRECT_SOLVE；PC 属于外层 fine 求解。
            base.solution_type = QudaSolutionType.QUDA_MAT_SOLUTION
            base.solve_type = QudaSolveType.QUDA_DIRECT_SOLVE
            base.gcrNkrylov = 16
            # QUDA MG 的 fine/smoother 使用 sloppy 精度；resident 主场仍保留 double。
            for prefix in ("", "clover_"):
                setattr(base, prefix + "cpu_prec", QudaPrecision.QUDA_DOUBLE_PRECISION)
                setattr(base, prefix + "cuda_prec", QudaPrecision.QUDA_DOUBLE_PRECISION)
                for suffix in ("sloppy", "refinement_sloppy", "precondition", "eigensolver"):
                    setattr(base, prefix + "cuda_prec_" + suffix, QudaPrecision.QUDA_SINGLE_PRECISION)
            mg_param.n_vec = [cfg.n_vec] * QUDA_MAX_MG_LEVEL
            # 首轮与 Chroma bridge 对齐：禁用所有 MMA，采用两次 block 正交化。
            mg_param.n_block_ortho = [2] * QUDA_MAX_MG_LEVEL
            mg_param.block_ortho_two_pass = [QudaBoolean.QUDA_BOOLEAN_TRUE] * QUDA_MAX_MG_LEVEL
            mg_param.coarse_solver = [QudaInverterType.QUDA_GCR_INVERTER] * QUDA_MAX_MG_LEVEL
            mg_param.smoother_halo_precision = [QudaPrecision.QUDA_SINGLE_PRECISION] * QUDA_MAX_MG_LEVEL
            mg_param.thin_update_only = QudaBoolean.QUDA_BOOLEAN_FALSE
            mg_param.geo_block_size = [list(cfg.block_size) + [1] * (QUDA_MAX_DIM - 4)
                                       for unused in range(mg_param.n_level)]
            mg_param.setup_use_mma = [0] * QUDA_MAX_MG_LEVEL
            mg_param.dslash_use_mma = [0] * QUDA_MAX_MG_LEVEL
            mg_param.transfer_use_mma = [0] * QUDA_MAX_MG_LEVEL
            mg_param.precision_null = [QudaPrecision.QUDA_SINGLE_PRECISION] * QUDA_MAX_MG_LEVEL
            mg_param.setup_maxiter = [cfg.setup_maxiter] * QUDA_MAX_MG_LEVEL
            mg_param.setup_maxiter_refresh = [cfg.refresh_maxiter] * QUDA_MAX_MG_LEVEL
            mg_param.setup_tol = [cfg.setup_tol] * QUDA_MAX_MG_LEVEL
            entry.mg = Multigrid(mg_param, base)
            _set_mg_setup_seed(cfg)
            self._mg.new()
            self._pool_event("setup", setup_seed=cfg.setup_seed, seed_backend=cfg.seed_backend)
        elif self._mg_dirty or self._refresh_pending and not self._suspended:
            # 每个 gauge 版本都更新 fine/coarse；常规更新不重新迭代 null vectors。
            _set_mg_setup_seed(self.config.multigrid)
            refresh_iterations = self.config.multigrid.refresh_maxiter if self._refresh_pending and not self._suspended else 0
            with parameter_scope(self._mg.param, setup_maxiter_refresh=[refresh_iterations] * QUDA_MAX_MG_LEVEL):
                self._mg.update(False)
            self._pool_event("refresh" if refresh_iterations else "update",
                             refresh_iterations=refresh_iterations, basis_frozen=bool(self._suspended))
        entry.dirty = False
        entry.resident_epoch = _RESIDENT_EPOCH
        _MG_ACTIVE = entry
        self._unbound_dirty = False
        if not self._suspended:
            entry.refresh_pending = False
        return self._mg.instance


    def _residual(self, out, rhs, param):
        _, residual_norm = self._normal_residual(out, rhs, param)
        return residual_norm


    def _normal_residual(self, out, rhs, param):
        """返回原始正规方程残差场及其相对范数，供验收和迭代修正共用。"""
        tmp = LatticeFermion(self.latt_info)
        residual = LatticeFermion(self.latt_info)
        # MatQuda 的 MATPC 与 invertQuda 的 KAPPA 归一化一致；不额外乘 2*kappa。
        with parameter_scope(param, solution_type=QudaSolutionType.QUDA_MATPC_SOLUTION,
                             dagger=QudaDagType.QUDA_DAG_NO, cuda_prec=QudaPrecision.QUDA_DOUBLE_PRECISION,
                             clover_cuda_prec=QudaPrecision.QUDA_DOUBLE_PRECISION):
            MatQuda(tmp.even_ptr, out.even_ptr, param)
            param.dagger = QudaDagType.QUDA_DAG_YES
            MatQuda(residual.even_ptr, tmp.even_ptr, param)
        residual.even = residual.even - rhs.even
        b2 = rhs.even.norm2()
        r2 = residual.even.norm2()
        # 非有限/负范数不能被当作零 RHS；返回 NaN 交给统一门禁和已有重试逻辑拒绝。
        if not math.isfinite(b2) or not math.isfinite(r2) or b2 < 0.0 or r2 < 0.0:
            return residual, float("nan")
        relative = math.sqrt(r2 / b2) if b2 > 0.0 else math.sqrt(r2)
        return residual, relative


    def _validate_normal(self, out, rhs, param, *, scope=None, residual=None, target=None, enforce=True):
        """MG 始终检查原方程；开关只决定普通 CG 是否额外做显式残差核验。"""
        if residual is None and not _VALIDATION_ENABLED:
            return
        residual = self._residual(out, rhs, param) if residual is None else residual
        target = float(param.tol if target is None else target)
        converged = math.isfinite(residual) and math.isfinite(target) and target > 0 and residual <= target
        self.last_validation = dict(action=self.label, scope=scope or ("temporary" if self._suspended else self.scope),
                                    shift=float(param.mu), target=target, true_residual=residual,
                                    converged=converged, equation="original_normal")
        _emit_telemetry("normal_residual", **self.last_validation)
        if enforce and not converged:
            raise SolverValidationError(self.last_validation)
        return converged

    def solve(self, out, rhs, param, performance):
        self._used = True
        self.last_solve = None
        self.last_validation = None
        self._require_normal_pool_available(self._entry)
        if self.config.multigrid.enabled:
            self._require_normal_pool_available(_MG_POOL.get(self._pool_key(param)))
        if self._suspended and self.config != self._suspend_config:
            self._defer("configuration changed during temporary scope")
            raise RuntimeError("temporary solver configuration changed; normal basis preserved")
        if self.config.multigrid.enabled and (not self._suspended or self.config.multigrid.force_gradient):
            self._solve_mg(out, rhs, param, performance)
        else:
            if not self._suspended and self._entry is not None:
                self._destroy_mg()
            self.solve_plain(out, rhs, param, performance, scope=self.scope)

    def solve_plain(self, out, rhs, param, performance, *, compute_action=False, scope=None):
        self._used = True
        self.last_solve = None
        self.last_validation = None
        # MD 入口显式传 scope；冻结期间统一标记 temporary，采样/能量沿用默认分类。
        scope = "temporary" if self._suspended else (scope or ("action" if compute_action else "sample"))
        with parameter_scope(param, inv_type=QudaInverterType.QUDA_CG_INVERTER,
                             solution_type=QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION,
                             solve_type=QudaSolveType.QUDA_NORMOP_PC_SOLVE,
                             dagger=QudaDagType.QUDA_DAG_NO,
                             compute_action=int(compute_action), **_CHRONO_OFF):
            self._invert(out, rhs, param, performance, scope=scope)
            self._validate_normal(out, rhs, param, scope=scope)
        self.last_solve = dict(iterations=[int(param.iter)], multigrid_used=False, scope=scope)

    def _invert(self, out, rhs, param, performance, direction="MdagM", scope=None, *, physical_shift=None):
        invertQuda(out.even_ptr, rhs.even_ptr, param)
        performance()
        _emit_telemetry("solve", action=self.label, scope=scope or ("temporary" if self._suspended else self.scope),
                        direction=direction, shift=float(param.mu),
                        physical_shift=float(param.mu) if physical_shift is None else physical_shift,
                        iterations=int(param.iter), target=float(param.tol),
                        multigrid_used=param.solve_type == QudaSolveType.QUDA_DIRECT_PC_SOLVE,
                        fallback=False)

    def _invert_mg_adjoint(self, out, rhs, param, performance, parity):
        """symmetric PC 伴随相似变换；必须复核物理正 shift 的原始伴随方程。"""
        transformed_rhs = LatticeFermion(self.latt_info)
        transformed_solution = LatticeFermion(self.latt_info)
        workspace = LatticeFermion(self.latt_info)
        _mg_clover(workspace, rhs, param, parity, True)
        _mg_gamma5(transformed_rhs, workspace)
        physical_shift = float(param.mu)
        with parameter_scope(param, mu=-float(param.mu), dagger=QudaDagType.QUDA_DAG_NO, **_CHRONO_OFF):
            self._invert(transformed_solution, transformed_rhs, param, performance,
                         "Mdag", physical_shift=physical_shift)
        _mg_gamma5(workspace, transformed_solution)
        _mg_clover(out, workspace, param, parity, False)
        checked = LatticeFermion(self.latt_info)
        with parameter_scope(param, dagger=QudaDagType.QUDA_DAG_YES):
            MatQuda(checked.even_ptr, out.even_ptr, param)
        checked.even = checked.even - rhs.even
        r2, b2 = checked.even.norm2(), rhs.even.norm2()
        residual = (math.sqrt(r2 / b2) if math.isfinite(r2) and math.isfinite(b2)
                    and r2 >= 0 and b2 > 0 else float("nan"))
        if not math.isfinite(residual) or not math.isfinite(param.tol) or param.tol <= 0 or residual > param.tol:
            self.last_validation = dict(equation="original_adjoint", shift=float(param.mu),
                                        true_residual=residual, target=float(param.tol), converged=False)
            raise SolverValidationError(self.last_validation)

    def _direct_options(self, param):
        target = param.tol
        options = dict(inv_type=QudaInverterType.QUDA_GCR_INVERTER,
                       inv_type_precondition=QudaInverterType.QUDA_MG_INVERTER,
                       dslash_type_precondition=QudaDslashType.QUDA_CLOVER_WILSON_DSLASH,
                       solution_type=QudaSolutionType.QUDA_MATPC_SOLUTION,
                       solve_type=QudaSolveType.QUDA_DIRECT_PC_SOLVE,
                       compute_action=0, compute_true_res=1, gcrNkrylov=16,
                       tol=target * 0.1, tol_restart=target * 0.1,
                       reliable_delta=1e-5, tol_precondition=0.25, maxiter_precondition=16,
                       schwarz_type=QudaSchwarzType.QUDA_ADDITIVE_SCHWARZ,
                       precondition_cycle=1, omega=1.0,
                       preconditioner=param.preconditioner)
        for prefix in ("", "clover_"):
            options[prefix + "cuda_prec"] = QudaPrecision.QUDA_DOUBLE_PRECISION
            # 与 base 的全部 resident 精度一致，避免 checkGauge 释放 MG 引用的副本。
            for suffix in ("sloppy", "refinement_sloppy", "precondition", "eigensolver"):
                options[prefix + "cuda_prec_" + suffix] = QudaPrecision.QUDA_SINGLE_PRECISION
        return options


    def _solve_mg(self, out, rhs, param, performance):
        parity = _mg_adjoint_parity(param)
        validate_multigrid_topology(self.latt_info, self.config.multigrid)
        temporary = bool(self._suspended)
        target = float(param.tol)
        if not math.isfinite(target) or target <= 0:
            raise ValueError("MG residual target must be finite and positive")
        intermediate = LatticeFermion(self.latt_info)
        options = self._direct_options(param)
        for retry in range(1 if temporary else 2):
            self.last_validation = None
            iterations = []
            try:
                options["preconditioner"] = self._prepare_mg(param, rebuild=bool(retry))
                if rhs.even.norm2() == 0.0:
                    out.data = 0
                    self._validate_normal(out, rhs, param, residual=0.0, target=target)
                    self.last_solve = dict(iterations=[], true_residual=0.0, retry=retry, multigrid_used=True)
                    return
                with parameter_scope(param, **options, **_CHRONO_OFF):
                    self._invert_mg_adjoint(intermediate, rhs, param, performance, parity)
                    iterations.append(int(param.iter))
                    with parameter_scope(param, dagger=QudaDagType.QUDA_DAG_NO):
                        self._invert(out, intermediate, param, performance, "M")
                        iterations.append(int(param.iter))
                    residual = self._residual(out, rhs, param)
                    converged = self._validate_normal(out, rhs, param, residual=residual, target=target, enforce=False)
                self.last_solve = dict(directions=["Mdag", "M"], iterations=iterations,
                                       true_residual=residual, retry=retry, multigrid_used=True,
                                       converged=converged, setup_seed=self.config.multigrid.setup_seed)
                if not converged:
                    raise SolverValidationError(self.last_validation)
                # FG 高迭代只置 pending，退出后才允许普通 MD 消费 refresh。
                self._refresh_pending |= max(iterations) >= self.config.multigrid.refresh_threshold
                return
            except _TemporaryMGConflict:
                raise
            except Exception as exc:
                self._pool_event("solve_attempt_failure", retry=retry, fallback=False, error=str(exc))
                if temporary:
                    self._defer("temporary MG failure")
                    raise
                if retry:
                    self._discard_pool_mg("MG failure")
                    self._destroy_mg()
                    raise
