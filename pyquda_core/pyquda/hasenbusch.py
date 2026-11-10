"""Clover-Wilson Hasenbusch HMC 的通用构造接口。

该模块只负责把物理配置转换为 action 列表和嵌套 HMC。格点尺寸、边界条件、
组态路径和 trajectory 循环仍由调用方负责，因此同一接口可用于不同体积的组态。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Type

from .action import (
    CloverWilsonAction,
    CloverWilsonHasenbuschTwistAction,
    CloverWilsonLogDetAction,
    GaugeAction,
)
from .action.abstract import Action, FermionAction, LoopParam, RationalParam
from .field import LatticeInfo
from .hmc import HMC, Integrator


@dataclass(frozen=True)
class StoutHMCConfig:
    """HMC stout forward/reverse 的公共配置。``dir_ignore=3/4`` 分别表示 3D/4D。"""

    n_smear: int = 1
    rho: float = 0.125
    dir_ignore: int = 4
    anisotropy: float = 1.0

    def as_hmc_kwargs(self) -> dict:
        return {
            "stout_n_smear": self.n_smear,
            "stout_rho": self.rho,
            "stout_dir_ignore": self.dir_ignore,
            "stout_smear_anisotropy": self.anisotropy,
        }


@dataclass(frozen=True)
class CloverHasenbuschConfig:
    """两味 Clover-Wilson determinant 的 Hasenbusch 分解参数。

    ``shifts`` 必须从 0 开始。若有 N 个 shift 点，工厂会生成 N-1 个相邻
    determinant-ratio action，并用最后一个 shift 生成 cancel action。
    """

    mass: float
    shifts: Sequence[float]
    clover_csw: float
    tol: float = 1e-6
    md_tol: Optional[float] = None
    action_tol: Optional[float] = None
    maxiter: int = 1000
    n_flavor: int = 2
    matpc_type: str = "odd_odd"
    ratio_force_mode: str = "pc_quda_norm"
    cancel_force_mode: str = "pc_quda_norm"
    ratio_force_source: str = "phi"
    cancel_constdet: bool = True
    cancel_sample_scale: float = 1.0
    force_sign: float = 1.0
    force_coeff_scale: float = 1.0
    force_output_scale: float = 1.0
    force_dagger: str = "yes"
    sym_odd_clover_scale: float = 1.0
    # QUDA Hasenbusch twist mass 映射：Chroma mu -> QUDA mu/kappa。
    # ratio 和 cancel 使用同一个 TwistedShiftedLinOp，因此两端都需缩放。
    # None 表示按当前轻夸克质量自动计算 1/kappa。
    # 迁移注意：旧配置若已使用 QUDA 单位的 shifts，必须显式设 mu_scale=1.0。
    mu_scale: Optional[float] = None

    def validated_shifts(self) -> List[float]:
        shifts = [float(value) for value in self.shifts]
        if self.n_flavor != 2:
            raise ValueError("Clover Hasenbusch twist decomposition currently supports exactly two flavors")
        if len(shifts) < 2:
            raise ValueError("Hasenbusch shifts must contain at least two values")
        if abs(shifts[0]) > 1e-15:
            raise ValueError("Hasenbusch shifts must start at 0")
        if any(next_value <= value for value, next_value in zip(shifts, shifts[1:])):
            raise ValueError("Hasenbusch shifts must be strictly increasing")
        return shifts


@dataclass(frozen=True)
class CloverDeterminantConfig:
    """一个普通 Clover-Wilson determinant action 的声明式配置。

    ``rational_param`` 决定 flavor 的有理近似；质量和 flavor 数只在这里变化，
    不需要 runner 直接实例化 ``CloverWilsonAction``。
    """

    rational_param: RationalParam
    mass: float
    n_flavor: int
    tol: float
    maxiter: int
    clover_csw: float


@dataclass(frozen=True)
class CloverHMCConfig:
    """从配置直接构造完整 Clover HMC 所需的公共参数。

    体积和边界条件属于 ``LatticeInfo``；3D/4D 只由 ``stout`` 改变；
    flavor 结构由 ``ordinary_primary`` 和 ``extra_actions`` 描述。
    """

    gauge_loop_param: LoopParam
    beta: float
    ordinary_primary: Sequence[CloverDeterminantConfig]
    extra_actions: Sequence[CloverDeterminantConfig]
    outer_integrator_type: Type[Integrator]
    outer_steps: int
    inner_integrator_type: Optional[Type[Integrator]]
    inner_steps: Optional[int]
    stout: StoutHMCConfig
    use_hasenbusch: bool
    hasenbusch: Optional[CloverHasenbuschConfig] = None
    placement: str = "all_fermion_outer"


@dataclass
class CloverHasenbuschActions:
    """工厂生成的 logdet、ratio 链和末端 cancel action。"""

    logdet: CloverWilsonLogDetAction
    ratios: List[CloverWilsonHasenbuschTwistAction]
    cancel: CloverWilsonHasenbuschTwistAction

    @property
    def all(self) -> List[FermionAction]:
        return [self.logdet, *self.ratios, self.cancel]


@dataclass
class HMCBuildResult:
    """构造后的 HMC 及实际积分层级，供 runner 原样写入日志。"""

    hmc: HMC
    inner_monomials: List[Action]
    outer_monomials: List[Action]
    hasenbusch_actions: Optional[CloverHasenbuschActions]


def build_clover_hasenbusch_actions(
    latt_info: LatticeInfo,
    config: CloverHasenbuschConfig,
) -> CloverHasenbuschActions:
    """按 shift 链构造完整的两味 Clover Hasenbusch determinant 分解。"""

    shifts = config.validated_shifts()
    kappa = 1 / (2 * (config.mass + latt_info.Nd))
    mu_scale = 1 / kappa if config.mu_scale is None else config.mu_scale
    common = {
        "md_tol": config.md_tol,
        "action_tol": config.action_tol,
        "force_sign": config.force_sign,
        "force_coeff_scale": config.force_coeff_scale,
        "force_output_scale": config.force_output_scale,
        "sym_odd_clover_scale": config.sym_odd_clover_scale,
        "force_dagger": config.force_dagger,
        "matpc_type": config.matpc_type,
    }
    logdet = CloverWilsonLogDetAction(
        latt_info,
        config.mass,
        config.n_flavor,
        config.tol,
        config.maxiter,
        config.clover_csw,
    )
    ratios = [
        CloverWilsonHasenbuschTwistAction(
            latt_info,
            config.mass,
            mu_scale * shift,
            config.tol,
            config.maxiter,
            config.clover_csw,
            determinant_ratio=True,
            mu_prec=mu_scale * shifts[index + 1],
            ratio_force_source=config.ratio_force_source,
            ratio_force_mode=config.ratio_force_mode,
            label=f"has{index}_{shift:g}_to_{shifts[index + 1]:g}",
            **common,
        )
        for index, shift in enumerate(shifts[:-1])
    ]
    cancel = CloverWilsonHasenbuschTwistAction(
        latt_info,
        config.mass,
        mu_scale * shifts[-1],
        config.tol,
        config.maxiter,
        config.clover_csw,
        determinant_ratio=False,
        constdet=config.cancel_constdet,
        cancel_force_mode=config.cancel_force_mode,
        cancel_sample_scale=config.cancel_sample_scale,
        label="has_cancel",
        **common,
    )
    return CloverHasenbuschActions(logdet=logdet, ratios=ratios, cancel=cancel)


def build_clover_hmc(
    latt_info: LatticeInfo,
    gauge_action: Action,
    ordinary_primary_actions: Sequence[FermionAction],
    extra_actions: Sequence[FermionAction],
    outer_integrator: Integrator,
    inner_integrator: Optional[Integrator],
    stout: StoutHMCConfig,
    *,
    use_hasenbusch: bool,
    hasenbusch: Optional[CloverHasenbuschConfig] = None,
    placement: str = "all_fermion_outer",
) -> HMCBuildResult:
    """构造普通或 Hasenbusch HMC，并显式返回实际 monomial 布局。

    ``ordinary_primary_actions`` 是关闭开关时使用的主 determinant；开启开关后由
    Hasenbusch 分解替代。``extra_actions`` 始终保留，可表示一味 strange 或其他
    flavor，因此 flavor 结构不会进入通用 Hasenbusch 实现。
    """

    if use_hasenbusch:
        if hasenbusch is None:
            raise ValueError("use_hasenbusch=True requires CloverHasenbuschConfig")
        has_actions = build_clover_hasenbusch_actions(latt_info, hasenbusch)
        primary = has_actions.all
    else:
        if hasenbusch is not None:
            raise ValueError("Hasenbusch config was supplied while use_hasenbusch=False")
        has_actions = None
        primary = list(ordinary_primary_actions)

    fermion_actions = [*primary, *extra_actions]
    if placement == "all_fermion_outer":
        inner_monomials = [gauge_action]
        outer_monomials = fermion_actions
    elif placement == "cancel_gauge_inner":
        if not use_hasenbusch or has_actions is None:
            raise ValueError("cancel_gauge_inner placement requires Hasenbusch actions")
        # 对齐 Chroma 生产 XML：快速层只含 gauge 与末端 cancel，logdet/ratio/额外 flavor 在外层。
        inner_monomials = [gauge_action, has_actions.cancel]
        outer_monomials = [has_actions.logdet, *has_actions.ratios, *extra_actions]
    elif placement == "cancel_inner":
        if not use_hasenbusch or has_actions is None:
            raise ValueError("cancel_inner placement requires Hasenbusch actions")
        inner_monomials = [gauge_action, has_actions.logdet, has_actions.cancel]
        outer_monomials = [*has_actions.ratios, *extra_actions]
    elif placement == "all_outer":
        inner_monomials = []
        outer_monomials = [gauge_action, *fermion_actions]
    else:
        raise ValueError(f"unknown HMC placement: {placement}")

    kwargs = stout.as_hmc_kwargs()
    if placement == "all_outer":
        hmc = HMC(latt_info, outer_monomials, outer_integrator, **kwargs)
    else:
        if inner_integrator is None:
            raise ValueError("nested HMC placement requires an inner integrator")
        hmc_inner = HMC(latt_info, inner_monomials, inner_integrator, **kwargs)
        hmc = HMC(latt_info, outer_monomials, outer_integrator, hmc_inner, **kwargs)
    return HMCBuildResult(hmc, inner_monomials, outer_monomials, has_actions)


def build_clover_hmc_from_config(
    latt_info: LatticeInfo,
    config: CloverHMCConfig,
) -> HMCBuildResult:
    """把声明式配置转换为 action、积分器和嵌套 HMC。

    这是 runner 应调用的高层入口。runner 只需解析 LIME/JSON/CLI 并填写配置，
    不再复制 Gauge、普通 determinant 或 Hasenbusch monomial 的构造流程。
    """

    def determinant_actions(items: Sequence[CloverDeterminantConfig]) -> List[CloverWilsonAction]:
        return [
            CloverWilsonAction(
                latt_info,
                item.rational_param,
                item.mass,
                item.n_flavor,
                item.tol,
                item.maxiter,
                item.clover_csw,
            )
            for item in items
        ]

    if config.outer_steps <= 0:
        raise ValueError("outer_steps must be positive")
    if config.placement != "all_outer" and (
        config.inner_integrator_type is None or config.inner_steps is None or config.inner_steps <= 0
    ):
        raise ValueError("nested HMC requires a positive inner_steps value and inner integrator type")

    gauge_action = GaugeAction(latt_info, config.gauge_loop_param, config.beta)
    outer_integrator = config.outer_integrator_type(config.outer_steps)
    inner_integrator = (
        None
        if config.inner_integrator_type is None or config.inner_steps is None
        else config.inner_integrator_type(config.inner_steps)
    )
    return build_clover_hmc(
        latt_info,
        gauge_action,
        determinant_actions(config.ordinary_primary),
        determinant_actions(config.extra_actions),
        outer_integrator,
        inner_integrator,
        config.stout,
        use_hasenbusch=config.use_hasenbusch,
        hasenbusch=config.hasenbusch,
        placement=config.placement,
    )
