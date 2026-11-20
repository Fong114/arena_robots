import functools
import os
import sys
import typing
from pathlib import Path

import attrs
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory, get_package_share_path
from arena_simulation_setup.shared import Light
from arena_simulation_setup.tree import Identifier, PathView, SimplePathResolver
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.models import ModelWrapper
from arena_simulation_setup.utils.models.model_loader import ModelProvider_USD

from arena_robots import assembly as assembly_mod
from arena_robots import preset as preset_mod
from arena_robots import urdf_facts
from arena_robots.caps import MobileSpec, RobotCaps
from arena_robots.catalog import Catalog, render_effective_sensors
from arena_robots.Sensor import SensorSpec, fill_frames

_CATALOG = Catalog()


@attrs.frozen
class ControlSpec:
    """ros2_control wiring declared in ``model_params.yaml`` under ``control:``.

    Presence in the YAML opts the robot into the ros2_control path: an
    in-process controller_manager (hosted by the URDF's gz_ros2_control
    plugin, or a ros2_control_node under Isaac) whose ``controllers`` the
    task generator loads and activates at robot bring-up. Absence means the
    gazebo_native path (PosePublisher + pose_to_tf + bridged cmd_vel) runs instead.
    """

    mode: str
    controllers: tuple[str, ...]
    config: str | None = None
    cmd_vel_topic: str = "cmd_vel"
    odom_topic: str = "odom"

    @classmethod
    def from_dict(cls, data: typing.Mapping[str, typing.Any]) -> "ControlSpec":
        mode = str(data.get("mode", "gazebo_native"))
        controllers_raw = data.get("controllers", [])
        if not isinstance(controllers_raw, list):
            raise ValueError(f"control.controllers must be a list; got {type(controllers_raw).__name__}")
        controllers = tuple(str(c) for c in controllers_raw)
        config = data.get("config")
        return cls(
            mode=mode,
            controllers=controllers,
            config=str(config) if config is not None else None,
            cmd_vel_topic=str(data.get("cmd_vel_topic", "cmd_vel")),
            odom_topic=str(data.get("odom_topic", "odom")),
        )

    @property
    def is_ros2_control(self) -> bool:
        return self.mode == "ros2_control"


def _read_mapping(path: str) -> dict[str, typing.Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"control config at {path} must be a mapping; got {type(data).__name__}")
    return data


def _overlay(base: dict, top: dict) -> dict:
    return {**base, **{k: _overlay(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v for k, v in top.items()}}


class ModelParams(dict[str, typing.Any]):
    """Robot-wide identity (from ``model_params.yaml``) with typed accessors
    that delegate to the sibling ``caps.yaml`` for mobile primitives.

    Constructed via :meth:`from_yaml` which records the file path; the
    ``caps`` cached property resolves ``<path>.parent`` so every
    ``ModelParams`` that came from disk can see its robot's caps.yaml without
    external wiring."""

    _path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ModelParams':
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Top-level structure in {path} must be a mapping")
        inst = cls(data)
        inst._path = path
        return inst

    @functools.cached_property
    def caps(self) -> RobotCaps:
        """Lazy view over ``<robot_dir>/caps.yaml``. Empty if this ``ModelParams``
        wasn't loaded from disk or the file doesn't exist."""
        robot_dir = self._path.parent if self._path is not None else Path('/dev/null')
        return RobotCaps(robot_dir=robot_dir)

    def _mobile(self) -> MobileSpec | None:
        if 'mobile' not in self.caps.available:
            return None
        return self.caps.mobile

    @property
    def base_frame(self) -> str:
        return str(self.get('base_frame', self.get('robot_base_frame', 'base_link')))

    @property
    def odom_frame(self) -> str:
        m = self._mobile()
        if m is not None:
            return m.odom_frame
        return self.get('robot_odom_frame', 'odom')

    @property
    def z_offset(self) -> float:
        return float(self.get('z_offset', 0.0))

    @property
    def control(self) -> ControlSpec | None:
        """Typed view of the ``control:`` block in model_params.yaml. ``None``
        when absent (gazebo_native pipeline)."""
        raw = self.get('control')
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"model_params 'control' must be a mapping; got {type(raw).__name__}")
        return ControlSpec.from_dict(raw)

    @property
    def sensors(self) -> list["SensorSpec"]:
        raw = self.get('sensors', [])
        if not isinstance(raw, list):
            raise ValueError(f"model_params 'sensors' must be a list; got {type(raw).__name__}")
        out: list[SensorSpec] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"model_params 'sensors[{i}]' must be a mapping; got {type(entry).__name__}")
            missing = {'name', 'type', 'topic'} - set(entry)
            if missing:
                raise ValueError(f"model_params 'sensors[{i}]' missing required keys: {sorted(missing)}")
            out.append(
                SensorSpec(
                    name=str(entry['name']),
                    type=str(entry['type']),
                    topic=str(entry['topic']),
                    frame=str(entry.get('frame', '')),
                    sensor=str(entry['sensor']) if 'sensor' in entry else None,
                )
            )
        return out

    @property
    def lights(self) -> list[Light]:
        """Lights the robot carries, each on a frame of the robot and intrinsic, so the world keeps its default lighting."""
        raw = self.get('lights', [])
        if not isinstance(raw, list):
            raise ValueError(f"model_params 'lights' must be a list; got {type(raw).__name__}")
        out: list[Light] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"model_params 'lights[{i}]' must be a mapping; got {type(entry).__name__}")
            if 'frame' not in entry:
                raise ValueError(f"model_params 'lights[{i}]' needs a frame of the robot")
            light = converter.structure({**entry, 'intrinsic': True}, Light)
            if any(light.name == other.name for other in out):
                raise ValueError(f"model_params 'lights' declares {light.name!r} twice")
            out.append(light)
        return out

    @property
    def priority(self) -> float:
        """Rank for `robot: auto`, lowest wins. Undeclared sorts after every explicit value."""
        return float(self.get('priority', float('inf')))

    @property
    def capabilities(self) -> list[dict[str, typing.Any]]:
        """Structured multi-adapter declaration as a list of dicts."""
        raw = self.get('capabilities', [])
        if not isinstance(raw, list):
            raise ValueError(f"model_params 'capabilities' must be a list; got {type(raw).__name__}")
        return [dict(entry) for entry in raw]


class RobotView(PathView):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._cached_params: ModelParams | None = None
        self._cached_control: dict | None = None
        self._cached_assembly: assembly_mod.AssemblySpec | None = None
        self._assembly_loaded: bool = False

    @property
    def is_definition(self) -> bool:
        """Whether the directory holds a robot definition, a ``model_params.yaml``."""
        return (self.path / 'model_params.yaml').is_file()

    @property
    def preset(self) -> preset_mod.Preset | None:
        """The directory's ``preset.yaml``, ``None`` for a robot that is no preset."""
        path = self.path / preset_mod.FILE_NAME
        return preset_mod.Preset.read(path) if path.is_file() else None

    @property
    def caps(self) -> RobotCaps:
        """Lazy view over robots/<name>/caps.yaml, equivalent to
        ``self.model_params.caps``, exposed directly on ``RobotView`` for
        readability."""
        return self.model_params.caps

    @property
    def mobile(self) -> MobileSpec | None:
        """The robot's mobile cap, or ``None`` if it doesn't advertise ``mobile``.

        Shortcut for the common ``caps.mobile`` access that returns ``None``
        honestly instead of raising ``FileNotFoundError`` when the cap is absent.
        """
        if 'mobile' not in self.caps.available:
            return None
        return self.caps.mobile

    @property
    def model_params(self) -> ModelParams:
        if self._cached_params is None:
            path = self.path / 'model_params.yaml'
            if not path.is_file():
                raise FileNotFoundError(f"model_params.yaml not found for robot '{self.name}' at {path}")
            self._cached_params = ModelParams.from_yaml(path)
            self._cached_params.caps.view = self
        return self._cached_params

    @property
    def control(self) -> dict:
        if self._cached_control is None:
            control_path = self.path / 'control.yaml'
            if not control_path.is_file():
                raise FileNotFoundError(f"control.yaml not found for robot '{self.name}' at {control_path}")
            with open(control_path) as f:
                mapping = yaml.safe_load(f)
                if not isinstance(mapping, dict):
                    raise ValueError(f"Control file {control_path} must contain a dictionary at the top level.")
                self._cached_control = mapping
        return self._cached_control

    def control_config(self, sim: str | None = None) -> dict[str, typing.Any]:
        """The ``control.config`` file as a mapping, overlaid key by key with its ``<name>.<sim>.yaml`` sibling when ``sim`` is given and that file exists."""
        control = self.model_params.control
        if control is None or control.config is None:
            raise FileNotFoundError(f"robot '{self.name}' declares no control.config")
        uri = control.config
        if uri.startswith("package://"):
            pkg, _, sub = uri[len("package://") :].partition('/')
            try:
                src_path = os.path.join(get_package_share_directory(pkg), sub)
            except PackageNotFoundError as e:
                raise FileNotFoundError(f"control.config package '{pkg}' not found") from e
        else:
            src_path = uri

        data = _read_mapping(src_path)
        if sim is None:
            return data
        root, ext = os.path.splitext(src_path)
        overlay_path = f"{root}.{sim}{ext}"
        if not os.path.isfile(overlay_path):
            return data
        return _overlay(data, _read_mapping(overlay_path))

    def drive_controller(self, sim: str | None = None) -> tuple[str, str, dict[str, typing.Any]] | None:
        """Name, type and parameters of the drive controller in ``control_config(sim)``, ``None`` for a robot without one."""
        control = self.model_params.control
        if control is None or control.config is None:
            return None
        return urdf_facts.drive_controller(self.control_config(sim))

    def drive_geometry(self, urdf: str, sim: str | None = None) -> urdf_facts.DriveGeometry:
        """The drive controller's geometry parameters measured in ``urdf``, empty for a robot without a drive controller."""
        controller = self.drive_controller(sim)
        return {} if controller is None else urdf_facts.drive_geometry(urdf, controller[1], controller[2])

    @property
    def bringup_launch(self) -> Path | None:
        """``launch/bringup.launch.py`` of the robot directory, ``None`` without one."""
        path = self.path / 'launch' / 'bringup.launch.py'
        return path if path.is_file() else None

    @property
    def srdf_source(self) -> Path | None:
        """The robot's own residual SRDF xacro fragment, ``None`` without one."""
        path = self.path / 'srdf' / f'{self.name}.srdf.xacro'
        return path if path.is_file() else None

    @property
    def assembly(self) -> assembly_mod.AssemblySpec | None:
        """Lazy view over robots/<name>/assembly.yaml, or ``None`` for robots that don't
        declare one (they resolve directly from model_params.yaml instead)."""
        if not self._assembly_loaded:
            path = self.path / 'assembly.yaml'
            if path.is_file():
                with open(path) as f:
                    data = yaml.safe_load(f)
                self._cached_assembly = assembly_mod.AssemblySpec.parse(data if isinstance(data, dict) else {})
            else:
                self._cached_assembly = None
            self._assembly_loaded = True
        return self._cached_assembly

    def _resolved(self, parts: dict[str, list[assembly_mod.RequestPart]]) -> assembly_mod.ResolvedAssembly | None:
        """Shared resolve step for :meth:`effective_sensors`/:meth:`effective_caps`.
        ``None`` iff the robot has no assembly.yaml."""
        spec = self.assembly
        if spec is None:
            return None
        return assembly_mod.resolve(spec, parts)

    def effective_sensors(self, parts: dict[str, list[assembly_mod.RequestPart]], *, frames: dict[str, str] | None = None) -> list[SensorSpec]:
        """The robot's full effective ``SensorSpec`` list for a parts request (``{}``
        for pure defaults). Robots with no assembly.yaml (``assembly`` is ``None``)
        return ``model_params.sensors`` with frames they omit read from the canonical URDF. Robots with an assembly.yaml
        resolve ``parts`` against the declared assembly and render it through the
        shared component catalog. No per-request caching: ``RobotView`` is a
        per-model singleton, but resolution is per-request.

        Consistency guard: warns on stderr, once per call, if ``model_params.sensors``
        declares a name absent from the assembly's own DEFAULTS render, i.e. the
        assembly's defaults must render every sensor model_params.yaml declares."""
        resolved = self._resolved(parts)
        if resolved is None:
            from arena_robots import served_urdf

            return fill_frames(self.model_params.sensors, lambda: served_urdf.canonical(self, parts, frames))
        assert self.assembly is not None
        resolved = assembly_mod.apply_frame_overrides(resolved, frames or {})
        prefix = self.assembly.prefix
        rendered = render_effective_sensors(resolved, _CATALOG, prefix=prefix)
        defaults_rendered = rendered if not parts else render_effective_sensors(assembly_mod.resolve(self.assembly, {}), _CATALOG, prefix=prefix)
        missing = {s.name for s in self.model_params.sensors} - {s.name for s in defaults_rendered}
        if missing:
            print(
                f"assembly.yaml incomplete? model_params declares sensors absent from assembly defaults: {sorted(missing)}",
                file=sys.stderr,
            )
        return rendered

    def effective_caps(
        self,
        parts: dict[str, list[assembly_mod.RequestPart]],
        *,
        frames: dict[str, str] | None = None,
        caps: dict[str, object] | None = None,
    ) -> RobotCaps:
        """The robot's effective :class:`RobotCaps` for a parts request (``{}`` for
        pure defaults), mirroring :meth:`effective_sensors`, with the instance's cap
        overrides ``caps`` applied, backed by this view so URDF-derived facts resolve. Robots
        with an assembly.yaml get ``resolved``/``catalog`` set so `.available`/instances see
        placement-derived caps."""
        resolved = self._resolved(parts)
        if resolved is None:
            return RobotCaps(self.caps.robot_dir, overrides=dict(caps or {}), view=self, parts=parts)
        assert self.assembly is not None
        resolved = assembly_mod.apply_frame_overrides(resolved, frames or {})
        return RobotCaps(self.caps.robot_dir, resolved=resolved, catalog=_CATALOG, prefix=self.assembly.prefix, overrides=dict(caps or {}), view=self, parts=parts, frames=dict(frames or {}))

    def power_profile(self, caps: RobotCaps) -> dict[str, float | int] | None:
        """Power-publisher parameters from the instance's effective power and body facts plus the resolved component draw. ``None`` when unpowered."""
        power = caps.power
        assembly_power = self.assembly.power if self.assembly is not None else None
        if power is None and assembly_power is None:
            return None

        raw = power.raw if power is not None else {}
        ps = raw.get('power_system') or {}
        static = raw.get('static_power_w', 0.0)
        static_w = float(static.get('compute_core', 0.0)) + float(static.get('idle_motors', 0.0)) if isinstance(static, dict) else float(static)
        efficiency = float(ps.get('global_drivetrain_efficiency', ps.get('drivetrain_efficiency', 1.0)))
        heating_ch = float(ps.get('heating_coefficient_ch', 0.0))
        battery_wh = float(ps.get('battery_capacity_wh', 0.0))
        damping = float(ps.get('drivetrain_damping', 0.0))
        friction = float(ps.get('drivetrain_friction', 0.0))
        crr = float(ps.get('rolling_resistance_crr', 0.0))
        controller = self.drive_controller()
        geometry = self.drive_geometry(caps.rendered_urdf())
        radius = geometry.get('wheel_radius', geometry.get('kinematics.wheels_radius', 0.0))
        wheel_r = radius if isinstance(radius, float) else 0.0
        num_wheels = len(urdf_facts.wheel_joints(controller[1], controller[2])) if controller is not None else 0
        mass_kg = float(caps.body.mass or 0.0)

        if assembly_power is not None:
            spec = assembly_power
            if static_w == 0.0:
                static_w = spec.compute_core_w + spec.idle_motors_w
            if efficiency == 1.0 and spec.drivetrain_efficiency != 1.0:
                efficiency = spec.drivetrain_efficiency
            if heating_ch == 0.0:
                heating_ch = spec.heating_coefficient_ch
            if battery_wh == 0.0:
                battery_wh = spec.battery_capacity_wh
            if damping == 0.0:
                damping = spec.drivetrain_damping
            if friction == 0.0:
                friction = spec.drivetrain_friction
            if crr == 0.0:
                crr = spec.rolling_resistance_crr

        if caps.resolved is not None:
            for placement in caps.resolved.placements:
                component = _CATALOG.get(placement.type, placement.variant)
                draw = component.power.get('static_power_w', 0.0)
                if not isinstance(draw, (int, float)):
                    raise ValueError(f"component {placement.type}/{placement.variant}: 'power.static_power_w' must be a number; got {type(draw).__name__}")
                static_w += float(draw)

        return {
            'static_power_w': static_w,
            'drivetrain_efficiency': efficiency,
            'heating_coefficient_ch': heating_ch,
            'battery_capacity_wh': battery_wh,
            'drivetrain_damping': damping,
            'drivetrain_friction': friction,
            'rolling_resistance_crr': crr,
            'robot_mass_kg': mass_kg,
            'wheel_radius_m': wheel_r,
            'num_wheels': num_wheels,
        }

    @property
    def model(self) -> ModelWrapper:
        return ModelWrapper(
            self.name,
            ModelProvider_USD.asdict(self.path, self.name),
        )


@attrs.define(eq=False, hash=False)
class RobotIdentifier(Identifier[RobotView]):
    def load(self, path: Path, /, **kwargs: object) -> RobotView:
        del kwargs  # unused
        return RobotView(path)


RobotIdentifier.use(SimplePathResolver(RobotIdentifier, get_package_share_path('arena_robots') / 'robots'))
