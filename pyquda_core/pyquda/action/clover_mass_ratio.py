"""真实 bare mass 的 clover 质量比作用量，不包含独立的 clover logdet。

输入为已初始化的各向同性四维晶格、两个质量、raw clover_csw 及求解参数；
输出为伪费米子、作用量、link force 或动量增量，不初始化 GPU、不生成文件。
使用 CloverWilsonScalarMassRatioAction(..., solver_accel=None) 选择双精度 CG；
显式加速配置交给库内 helper，需其支持 solver_accel 和 _solve(..., md=...)。
采样和能量始终用 CG，只有力的分子求解可使用 MG。RNG adapter 仅绑定外层
sampleEta，传入已归一化的 eta；本类不再缩放噪声，也不扣能量常数。
HMC 调用 force_v2(1, force) 后统一施加积分步长并反传 stout；force(dt)
只投影当前算符 links，不执行 stout reverse。有限差分验证由 runner 负责。
"""

from contextlib import ExitStack, contextmanager
import math

from .abstract import FermionAction
from .clover_hasenbusch import CloverWilsonHasenbuschTwistAction
from ..enum_quda import QudaDagType, QudaPrecision
from ..field import LatticeGauge, LatticeMom


class CloverWilsonScalarMassRatioAction(FermionAction):
    """用两个 mu=0、symmetric odd_odd 算子构造 constdet 质量比。"""

    def __init__(
        self, li, mass, mass_precond, tol, maxiter, clover_csw, *,
        label="", md_tol=None, action_tol=None, solver_accel=None,
    ):
        mass, mass_precond, clover_csw, tol = map(float, (mass, mass_precond, clover_csw, tol))
        md_tol = tol if md_tol is None else float(md_tol)
        action_tol = tol if action_tol is None else float(action_tol)
        if not all(math.isfinite(v) for v in (mass, mass_precond, clover_csw, tol, md_tol, action_tol)):
            raise ValueError("mass/clover/tolerance must be finite")
        if li.Nd != 4 or li.anisotropy != 1.0:
            raise ValueError("scalar ratio requires isotropic 4D")
        if min(mass, mass_precond) <= -4 or clover_csw <= 0:
            raise ValueError("require mass > -4 and raw clover_csw > 0")
        if not all(0 < value < 1 for value in (tol, md_tol, action_tol)):
            raise ValueError("require 0 < tol/md_tol/action_tol < 1")
        if not math.isfinite(float(maxiter)) or int(maxiter) != maxiter or maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        self.mass, self.mass_precond = mass, mass_precond
        self.clover_csw = clover_csw
        self.md_tol, self.action_tol = md_tol, action_tol
        self.label = label or f"scalar_ratio({mass},{mass_precond})"
        # 字典补齐默认精度；已解析的不可变配置对象交给 helper 原样处理。
        self._solver_config = dict(solver_accel) if isinstance(solver_accel, dict) else solver_accel
        if isinstance(self._solver_config, dict):
            self._solver_config.setdefault("plain_precision", "double")
        self.numer = self._new_helper(li, mass, tol, int(maxiter), "numer")
        self.denom = self._new_helper(li, mass_precond, tol, int(maxiter), "denom")
        super().__init__(li, self.numer.dirac)
        managers = tuple(getattr(helper, "solver_accel", None) for helper in (self.numer, self.denom))
        if any(manager is not None for manager in managers):
            if any(manager is None for manager in managers):
                raise RuntimeError("both mass helpers must expose solver_accel")
            self.solver_accel = _ScalarSolverManagers(*managers)
        else:
            if self._solver_config is not None:
                raise RuntimeError("configured acceleration requires helper.solver_accel")
            self.solver_accel = None
        # adapter/HMC 只识别外层 action；两条质量路径共享分子的伪费米子缓冲。
        self.eta, self.phi = self.numer.eta, self.numer.phi
        self.rhs, self.quark = self.numer.rhs, self.numer.quark
        self.y, self.source = self.numer.y, self.numer.source

    def _new_helper(self, li, mass, tol, maxiter, suffix):
        # 默认 CG 兼容尚无加速接口的 helper；显式配置不做静默降级。
        config = dict(self._solver_config) if isinstance(self._solver_config, dict) else self._solver_config
        kwargs = {} if config is None else {"solver_accel": config}
        helper = CloverWilsonHasenbuschTwistAction(
            li, mass, 0.0, tol, maxiter, self.clover_csw,
            md_tol=self.md_tol, action_tol=self.action_tol,
            constdet=True, determinant_ratio=False, matpc_type="odd_odd",
            cancel_force_mode="pc_quda_norm", cancel_sample_scale=1.0,
            label=f"{self.label}:{suffix}", **kwargs,
        )
        if self._solver_config is None:
            precision = QudaPrecision.QUDA_DOUBLE_PRECISION
            helper.dirac.setPrecision(
                cuda=precision, sloppy=precision, refinement_sloppy=precision,
                precondition=precision, eigensolver=precision,
            )
        return helper

    def _activate(self, helper):
        # resident clover 是进程全局状态，切换真实质量后必须重建。
        if helper.mu != 0.0 or helper.invert_param.mu != 0.0:
            raise RuntimeError("scalar mass ratio requires both helpers at mu=0")
        helper.md_tol, helper.action_tol = self.md_tol, self.action_tol
        helper.invert_param.compute_clover = 1
        helper.invert_param.compute_clover_inverse = 1
        helper.updateClover()

    def updateClover(self):
        self._activate(self.numer)

    def _active_parity(self):
        # 物理 odd 子格使用 QUDA MatPC 的首个存储槽，由 adapter 完成搬运。
        return self.numer._active_parity()

    def setVerbosity(self, verbosity):
        for helper in (self.numer, self.denom):
            helper.setVerbosity(verbosity)

    def sample(self):
        """仅刷新一次 eta；依次执行 A 的共轭、B 正规方程求解和 B 乘法。"""
        if self.solver_accel is not None:
            self.solver_accel.invalidate("scalar sample")
        self.sampleEta()
        self.eta.data[1] = 0
        self._activate(self.numer)
        self.numer._mat(self.rhs, self.eta, QudaDagType.QUDA_DAG_YES, 0.0)
        self._activate(self.denom)
        # 不传 md=True，刷新保持 CG；md_tol 仍由 helper 选择。
        self.denom._solve(self.rhs, 0.0, compute_action=False)
        self.denom._mat(self.phi, self.denom.quark[0], QudaDagType.QUDA_DAG_NO, 0.0)
        self.phi.data[1] = 0
        self.source[0] = self.phi

    def _solution(self, *, compute_action):
        self._activate(self.denom)
        self.denom._mat(self.rhs, self.phi, QudaDagType.QUDA_DAG_YES, 0.0)
        self._activate(self.numer)
        kwargs = {} if self.solver_accel is None else {"md": not compute_action}
        self.numer._solve(self.rhs, 0.0, compute_action=compute_action, **kwargs)

    def action(self):
        self._solution(compute_action=True)
        # helper 返回的正规方程作用量即质量比能量；不额外扣常数或 logdet。
        return float(self.numer.invert_param.action[0])

    def force_v2(self, dt, force=None):
        """累加 dt 倍的独立 link force；不缩放调用方已有贡献。"""
        dt = float(dt)
        if not math.isfinite(dt):
            raise ValueError("dt must be finite")
        if force is None:
            force = LatticeGauge(self.latt_info)
            force.data = 0
        if not isinstance(force, LatticeGauge):
            raise TypeError("force_v2 requires LatticeGauge, not LatticeMom")
        self._solution(compute_action=False)
        self.numer._mat(self.y, self.quark[0], QudaDagType.QUDA_DAG_NO, 0.0)
        contribution = LatticeGauge(self.latt_info)
        contribution.data = 0
        x = self.quark[0]
        # helper 计算算子导数；取负作用量导数得到 B 的两负项及 A 的两正项。
        terms = (
            (self.denom, self.phi, x, QudaDagType.QUDA_DAG_NO, -1.0, "B(phi,x)"),
            (self.denom, x, self.phi, QudaDagType.QUDA_DAG_YES, -1.0, "Bdag(x,phi)"),
            (self.numer, x, self.y, QudaDagType.QUDA_DAG_YES, 1.0, "Adag(x,y)"),
            (self.numer, self.y, x, QudaDagType.QUDA_DAG_NO, 1.0, "A(y,x)"),
        )
        active = None
        # helper 会记录展开项；每次只保留当前力，避免长期积分累积诊断记录。
        for helper in (self.numer, self.denom):
            helper.force_term_history.clear()
        for helper, left, right, dagger, sign, name in terms:
            if helper is not active:
                self._activate(helper)
                active = helper
            param = helper.gauge_param
            saved = (param.use_resident_force, param.return_result_force)
            param.use_resident_force, param.return_result_force = 0, 1
            try:
                # 显式使用真实 MatPC 归一化，不沿用旧探针的 clover_csw/kappa。
                helper._add_sym_oddodd_quda_norm_deriv_force(
                    contribution, left, right, dagger, sign, name, norm_scale=1.0,
                )
            finally:
                param.use_resident_force, param.return_result_force = saved
        # 四项全部成功才提交，异常时调用方的累计力保持原值。
        force.data += dt * contribution.data
        return force

    def force(self, dt, mom=None):
        """投影当前算符 links 的动量增量，乘一次 dt；不负责 stout reverse。"""
        dt = float(dt)
        if not math.isfinite(dt):
            raise ValueError("dt must be finite")
        if mom is not None and not isinstance(mom, LatticeMom):
            raise TypeError("force requires LatticeMom; use force_v2 for link force")
        from ..quda import saveGaugeQuda

        links = LatticeGauge(self.latt_info)
        saveGaugeQuda(links.data_ptrs, self.gauge_param)
        link_force = self.force_v2(1.0)
        xp = _array_module(links.data)
        uf = xp.einsum("...ab,...bc->...ac", links.data, link_force.data)
        ah = (uf - uf.conj().swapaxes(-1, -2)) / 2
        ah -= (ah.trace(axis1=-2, axis2=-1) / 3)[..., None, None] * xp.eye(3)
        delta = LatticeMom(self.latt_info)
        delta.data = 0
        # MILC 动量采用十实数布局：三个非对角复数、三个虚对角及一个填充槽。
        for index, (row, col) in enumerate(((0, 1), (0, 2), (1, 2))):
            delta.data[..., 2 * index] = dt * ah[..., row, col].real
            delta.data[..., 2 * index + 1] = dt * ah[..., row, col].imag
        for index in range(3):
            delta.data[..., 6 + index] = dt * ah[..., index, index].imag
        if mom is None:
            return delta
        mom.data += delta.data
        return mom


def _array_module(array):
    if hasattr(array, "__cuda_array_interface__"):
        import cupy
        return cupy
    import numpy
    return numpy


class _ScalarSolverManagers:
    """将 HMC 的算符上下文、FG 暂停、失效和释放同步到两个质量 helper。"""

    def __init__(self, *managers):
        self.managers = managers

    @property
    def operator_context(self):
        return self.managers[0].operator_context

    @operator_context.setter
    def operator_context(self, value):
        for manager in self.managers:
            manager.operator_context = value

    def configure(self, config=None):
        # HMC 只看到外层聚合对象，构造后的配置必须覆盖两个真实质量。
        for manager in self.managers:
            manager.configure(config)

    def invalidate(self, reason="operator change", **kwargs):
        for manager in self.managers:
            manager.invalidate(reason, **kwargs)

    @contextmanager
    def suspended(self):
        with ExitStack() as stack:
            for manager in self.managers:
                stack.enter_context(manager.suspended())
            yield

    def close(self):
        for manager in self.managers:
            manager.close()


# 兼容旧调用名称；实现和依赖均留在库内，不导入实验目录。
ScalarMassRatioAction = CloverWilsonScalarMassRatioAction
