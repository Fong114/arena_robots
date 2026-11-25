# Adding a robot

A robot is a directory under `arena_robots/arena_robots/robots/<name>/`. The
directory name is the robot's canonical identifier: task_generator, launch
files, and the `arena feature robots` CLI all look robots up by this name.

## Required files

```
<robot>/
├── model_params.yaml    # robot-wide identity (robot_model, base_frame, z_offset, control)
├── control.yaml         # ros2_control configuration
└── caps.yaml            # capabilities and robot facts, a robot without a `mobile` section is immobile
```

### `model_params.yaml`

Robot-wide identity. Cap data does not live here except for fields that apply
to any robot regardless of cap: `base_frame` (every robot has a base link),
`z_offset` (height of the model root when the robot rests on the ground, defaults to 0.0), `control` (the ros2_control wiring) and `lights` (lights the robot carries, defaults to `[]`).

```yaml
robot_model: my_robot       # required; matches the directory name
base_frame: base_link       # required; TF frame of the robot's base link
z_offset: 0.0635            # optional; metres from the ground to the model root with the robot at rest, spawn and teleport place it there
control:                    # optional; absent runs the gazebo_native pipeline
  mode: ros2_control
  config: package://arena_robots/robots/my_robot/control.yaml
lights:                     # optional; Light entries anchored on a frame of the robot
  - {name: headlight, fixture: spot, frame: base_link, offset: {x: 0.2, y: 0.0, z: 0.3, pitch: -1.42}, lumens: 1500}
```

Each `lights` entry takes the world `Light` schema with `frame` as its only anchor
([AUTHORING.md](../../../arena_simulation_setup/AUTHORING.md#lights)). The task generator
spawns it with the robot as `<robot>_<name>` on the robot's own frame, so it follows the
robot, is a `light` semantic entity (`arena env semantics <robot>_<name> lit=false`) and
leaves the world's default lighting on.

Parsed by [`arena_robots.Robot.ModelParams`](../arena_robots/Robot.py). Additional keys pass
through unchanged; nothing in-tree consumes them.

The sensors of a robot are the sensor blocks of its `assembly.yaml` components, `effective_sensors` renders them. `controllers`, `cmd_vel_topic` and `odom_topic` of `control` are read from the control config unless authored: the controllers are the `controller_manager` entries in file order, the topics are `<drive controller>/<cmd_vel|reference>` and `<drive controller>/<odom|odometry>` by controller type (a robot without a drive controller keeps `cmd_vel` and `odom`, `a1` and `go1` author `cmd_vel_stamped`). A robot without an `assembly.yaml` may still declare a `sensors:` list in `model_params.yaml`.

The sensors also drive the Nav2 costmap sources ([`arena_robots.nav2`](../arena_robots/nav2.py)).
The local costmap raytraces planar lidars in a 2D obstacle layer and runs multi-ring lidars
and depth cameras through a spatio-temporal voxel layer, which clears inside a frustum fitted
to the sensor's field of view. Field of view and range are read from the gz `<sensor>` element
of the canonical URDF (`RobotCaps.rendered_urdf()`) that the entry's `sensor:` key names, or its `name` when the key is absent
([`arena_robots.sensor_geometry`](../arena_robots/sensor_geometry.py)). A sensor without such
an element counts as planar when it only declares a `laserscan`. Otherwise it marks without
clearing and its obstacles expire after the layer's `voxel_decay`.

Every gz `<sensor>` sits on its `<gazebo reference>` link with an identity `<pose>`. A sensor
entry carries no `frame`: `effective_sensors` reads it from the canonical URDF, the `gz_frame_id`
of the gz `<sensor>` the entry's `sensor:` names, else that sensor's `<gazebo reference>` link
([`urdf_facts.sensor_frames`](../arena_robots/urdf_facts.py)). A camera's reference link also needs a REP-103
optical child link at zero offset, named `<reference>_optical_frame` with a trailing `_frame` of
the reference dropped. Isaac stamps image,
depth and point cloud messages in the optical frame, Gazebo and MuJoCo in the reference link,
and the voxel layer poses a camera's clearing frustum in the reference link.
`tests/test_sensor_geometry.py` checks this for every robot and camera component.

### `caps.yaml`: capabilities and robot facts

One file per robot. Each top-level key is a section, and a **section's presence
IS the advertisement**: an `arm:` key means this robot advertises `arm`. Derived
at load time as `robot_view.caps.available`, a `frozenset[str]` of section names.
`power`, `acoustics` and `drives` are fact blocks and never count as caps. The URDF is the
single source of everything it defines, caps.yaml holds everything else.

Adapters declare `requires: frozenset[str]` via `@BringupMeta.attach(requires=frozenset({...}))` on the `Bringup` subclass; the broker gates binding on `adapter.requires ⊆ robot.caps.available`. A cap must have a documented vocabulary entry before any adapter cites it in `requires`.

#### Active sections

| Section | Shape | Meaning |
|---|---|---|
| `mobile` | **flat** (singleton) | drivable chassis (wheeled / tracked / omni) |
| `arm` | **dict** (named instances) | serial manipulator with an IK-solvable tip |
| `lift` | **dict** (named instances) | prismatic vertical positioner (Ewellix column, torso lift, scissor), commanded by the `direct` lift adapter (`joint`, `controller`, optional `limits`) |
| `gripper` | **dict** (named instances) | end effector with one actuated joint (`joint`, `controller`, `positions`, `max_effort`, optional `arm`), commanded by the `direct` gripper adapter |
| `power` | flat (fact block) | parameters of the power model |
| `acoustics` | flat (fact block) | parameters of the acoustic emission model |
| `drives` | per joint (fact block) | servo of every joint ros2_control commands by position (`kp`, `kd`, `effort`, `command: position\|effort`), derived from the rendered URDF, author only what differs |

Reserved for future: `ptu_head`, `dual_arm`. Continuum arms either
discretize into the serial-chain shape or use a future `kind: continuum`
variant of the `arm` section.

#### Schema

A robot has exactly one mobile base by construction. All robot-physical
primitives of `mobile` live at its top level; adapter sub-blocks are reserved for
adapter-specific wiring only. `actions` is a top-level field. `laser`, `radius` and `is_holonomic` are not authored and `footprint` only where the URDF collision is smaller than the robot, see "Where each value is authored". The
optional `nav2:` sub-block carries nav2-only wiring such as
`global_observation` and `local_observation` (the cloud sources' `min_obstacle_height` and `max_obstacle_height`, 0.1 and 0.8 m by default). The global planner's settings come from the selected
planner config. `arm`, `lift` and `gripper` are always a dict of named instances: single-arm robots use a dict with one entry (by convention named `arm`), dual-arm uses two (e.g. `left`, `right`), with no single-instance shorthand.

```yaml
mobile:                       # required, flat
  source: https://...         # optional, the document the section's values come from, in every section
  odom_frame: odom
  drive: true                 # optional, default true, false for a base that only seats the robot
  min_turning_radius: 0.0     # optional, m, default 0.0 (turns in place)
  climbs: [stairs]            # optional, default []

  footprint_padding: 0.1      # optional
  inflation_radius: 0.9       # optional, default: outer radius of the padded footprint or radius, + 0.4
  polygons_dict:
    StopPolygon: {type: polygon, points: [...], action_type: stop}

  velocity_limits:
    linear: {min: -2.0, max: 2.0}
    angular: {min: -4.0, max: 4.0}
  acceleration_limits:
    linear: 3.0
    angular: 3.5

  actions:
    continuous: {linear: {min: -2.0, max: 2.0}, angular: {min: -4.0, max: 4.0}}   # optional, defaults to velocity_limits
    discrete: [{name: move_forward, linear: 0.3, angular: 0.0}, ...]

  limits:                     # optional, per axis, each axis defaults to planar's
    velocity: {linear: [-2.0, 2.0], angular: [-4.0, 4.0]}
    acceleration: {linear: 20.0, angular: 25.0}

  planar:
    velocity: {linear: [-2.0, 2.0], lateral: [0.0, 0.0], angular: [-4.0, 4.0]}
    acceleration: {linear: 20.0, lateral: .inf, angular: 25.0}
    gain: {linear: 1.0, lateral: 1.0, angular: 1.36}
    lag: 0.0
    traction: 7.85
    pivot: front              # front|rear: the wheel axle the body turns about, or a number in m, defaults to 0.0
    timeout: 0.25

  nav2:                       # optional, nav2-only wiring
    global_observation: {...}

  meshnav:                    # optional, meshnav-only wiring
    max_step: 0.2             # default 0.2 when climbs lists stairs, else 0.1
    max_slope: 1.3            # rad, default 1.3 when climbs lists stairs, else 0.35
    inscribed_radius: 0.21    # default inscribed circle of the padded footprint, else radius
    inflation_radius: 0.71    # default inscribed_radius + 0.5
    obstacles: true           # default true when the robot has a laserscan or pointcloud sensor
    robot_height: 0.7         # m, returns higher above the walking surface are ignored, default body.height plus 0.1
    min_obstacle_height: 0.1  # m, returns lower above the walking surface are ground
    obstacle_range: 11.0      # m from the sensor, default caps laser.range minus 1, else 9
    obstacle_decay: 1.0       # s a sensor's obstacles outlive its scan
    planner_frequency: 1.0    # Hz, MBF replans at it while a goal runs
    planner: dijkstra         # or cvp, MBF's fast marching planner
    resolution: 0.1           # grid spacing of the nav mesh
    params: {}                # merged over config/meshnav/meshnav.yaml

arm:                          # optional, dict of named instances (single-arm)
  arm:
    controller: arm_controller
    moveit:                   # adapter-specific; consumed by the moveit adapter
      planning_group: arm_manipulator
      srdf: {package: arena_robots, path: components/arm/ur/srdf/ur.srdf.xacro}
      args: {prefix: "", mount: arm, ur_type: ur5e}
    # base_link, tip_link, chain and the stow and ready named_poses come from the SRDF and URDF

lift:                         # optional, dict of named instances
  lift:
    joint: robot_lift_ewellix_lift_top_joint
    controller: lift_controller
    limits: [0.0, 0.9]        # joint range the lift adapter clamps to, optional, default the joint's URDF limit

gripper:                      # optional, dict of named instances
  gripper:
    arm: arm                  # key into the arm section
    joint: arm_gripper_joint
    controller: arm_gripper_controller
    moveit: {srdf: {package: arena_robots, path: components/gripper/robotiq_2f_85/srdf/robotiq_2f_85.srdf.xacro}, args: {prefix: "", mount: arm_gripper}}
    positions: {open: 0.0, close: 0.8}   # optional, default the SRDF group states open and close
    max_effort: 50.0          # optional, default the joint's URDF effort

power:                        # optional, fact block
  power_system:
    system_voltage_v: 24.0
    global_drivetrain_efficiency: 0.85
    heating_coefficient_ch: 0.05
    battery_capacity_wh: 270.0
    drivetrain_damping: 0.15
    drivetrain_friction: 0.65
    rolling_resistance_crr: 0.035
  joint_metrics: {topic: joint_states}
  static_power_w: {compute_core: 20.0, idle_motors: 15.0}

acoustics:                    # optional, fact block over config/acoustic_profile.yaml, author only what differs
  L_base_0: 40.0
  beta_0: 58.5
  beta_1: 20.0
  beta_2: 14.0
  beta_3: 0.0
  omega_ref: 10.2
  tau_ref: 2.0
  omega_deadband: 0.05
  omega_active: 0.50
  sigma_base: 0.8
  sigma_dynamic: 1.2
  sigma_no_effort: 2.0

drives:                       # optional, fact block over the rendered URDF, author only what differs
  FL_calf_joint: {kp: 40.0, kd: 1.0}
```
`drives.<joint>` covers every joint the ros2_control block commands by position. Its
defaults come from the rendered URDF: revolute and continuous joints kp 100 N m/rad per
N m of `<limit effort>` and kd kp/20, prismatic joints kp 8 N/m per N raised until
gravity sags the carried mass under 2 mm and kd critical, `effort` the URDF effort
(25 when unset). The served URDF carries them as `kp`, `kd` and `effort` params of the
ros2_control joint, MuJoCo and Isaac servo position joints with them at physics rate.
`command: effort` swaps the joint's position command interface for effort, which the
simulators apply raw, clamped to `effort`, after holding the spawn pose until the first
finite effort command. The robot's controllers must then command effort on that joint,
rendering raises otherwise. flatsim always commands by position.

`velocity_limits` and `acceleration_limits` flow into nav2 controller configs
via `Nav2KinematicsDerivedYAML` substitution; controller YAMLs reference them
as `${max_linear_vel:-default}` etc. To override a specific controller without
touching the `mobile` section, replace the substitution with a literal value in that
controller YAML. The same substitution emits `${motion_model}`, MPPI's motion
model (`Omni` when `is_holonomic`, `Ackermann` when `min_turning_radius` is
above 0, else `DiffDrive`), and for a car-like base `${turning_radius}`, the
`min_turning_radius` that MPPI's Ackermann constraint and the Smac Hybrid
planner turn at. The velocity smoother reads its lateral entries from
`max_lateral_vel`, `min_lateral_vel`, `lateral_acc` and `lateral_decel`, 0 for a
robot whose `velocity_limits` have no `lateral` (`lateral_acc` is the linear
acceleration when only the velocity is lateral). The HATEB footprint model takes
`radius` and `footprint`, its `wheelbase` is the front to rear distance of the
drive controller's wheel joints in the rendered URDF (1.0 without one), and
slam_toolbox maps out to `laser.range`.

`planar` is the body response `sim:=flatsim` drives the robot with, the same keys
for every robot: per-axis velocity bounds, acceleration and gain, a first-order
`lag`, a `traction` cap on planar acceleration, the `pivot` x the body turns
about (`front` or `rear` takes the largest or smallest wheel joint x of the drive controller from the URDF, a number is metres) and the command `timeout`. It is the drive's envelope, not the planner's,
so it can exceed `velocity_limits`.

`min_turning_radius` is the smallest radius in metres the base can turn on,
measured at `base_link`. Leave it out (0.0) for a base that turns in place:
differential, skid-steer and omni drives. Set it for a car-like (Ackermann)
base to `wheelbase / tan(max steering angle)`, with the steering angle of the
bicycle model, not of the inner wheel. A base with a radius above 0 cannot
spin, so its turn rate is bounded by `|forward speed| / min_turning_radius`.
It cannot be combined with `is_holonomic: true`. `f1tenth` is the car-like
robot. The nav2 adapter gives such a base the `default_no_spin` behavior tree
in place of `default`, whose spin recovery it cannot drive, unless
`robot.mobile.inter_planner` names another tree.

`limits` is the base's physical velocity and acceleration per axis, the one
source of the actuator limits: an axis it names replaces the same axis of
`planar.velocity` or `planar.acceleration`, and the effective cap carries
`limits` in their place. `velocity_limits` and `acceleration_limits` are
clamped to it on every axis, and a file whose planner limit exceeds it warns in
`robots_check`.

`climbs` lists the world stair kinds (`stairs`, `escalator`) the base can
drive through. Cross-level routing sends the robot through the boarding half of
a stair only when its kind is listed, and an explicit `via` naming a stair the
robot does not climb is an error. `go1` and `a1` climb stairs.

`meshnav:` sets the traversal limits of the meshnav adapter: the largest height
step and the steepest slope the base crosses, and the inflation around lethal
vertices. The obstacle layer reads `meshnav/obstacles`, published by the
`meshnav_obstacles` node from every range sensor (a sensor's point cloud when it
has one, else its scan) in the odom frame. A return is an obstacle when it stands
`min_obstacle_height` to `robot_height` above the highest mesh surface below it
and no mesh surface around it lies between `min_obstacle_height` under and
`max_step` over it, so floors, ramps, stair treads, risers and escalator belts are
ground on every level. Returns beyond `obstacle_range` or within the footprint's
outer radius are dropped. Velocity limits cap the mesh controller. Goals go to
MBF's Dijkstra mesh planner, `planner: cvp` picks the fast marching planner, whose
path back-tracking fails across some stair terraces.

`rosnav_rl.agent` names the agent directory under `ROSNAV_AGENTS_DIR` (else
`$ARENA_DATA_DIR/agents`). At launch the `rosnav_rl` bringup, and the `nav2`
bringup with `local_planner=rosnav_rl`, compare the continuous action ranges in
the agent's `training_config.yaml` with the instance's effective
`velocity_limits` (linear, angular, and lateral on a holonomic base) and log a
warning naming both ranges when they differ. A missing or unreadable agent
config logs that the check was skipped.

Bringups and the task server read caps through the instance's effective caps
(`Bringup.caps`: parts, frames and the request's cap overrides applied), so an
instance that overrides `mobile.velocity_limits` or a placed part's fields (`top=arm/fr3[stall_time=2.0]`) is
planned, launched and served with those values, while the model's `caps.yaml`
stays untouched.

The same effective caps carry the robot-wide facts as three more blocks. Authoring
stays in the files, every consumer reads the block:

| Block | Read from | Holds |
| --- | --- | --- |
| `body` | `model_params.yaml` and the rendered URDF | `base_frame`, `cmd_vel_topic` and `odom_topic` (from `control:`), `mass` (the summed inertial mass of the rendered URDF, placed parts included, which `RobotCaps.mass_split()` splits into the chassis `base`, the placed `components` and the placed `payload`) |
| `power` | the `power` section of `caps.yaml` | the section (the power model takes the mass from `body.mass`, the torque clamp from `acoustics`) |
| `acoustics` | the `acoustics` section of `caps.yaml` | the section as authored |

`body.*`, `power.*` and `acoustics.*` take cap overrides like any cap
(`power.power_system.rolling_resistance_crr=0.05`), except `body.mass`. `RobotCaps.provenance(path)` names where a value
comes from (`override`, `component`, `derived`, `urdf` or `file`) and `RobotCaps.snapshot()`
holds the effective caps with every leaf's source and a hash. `body.outline` (the
convex ground outline of the base's collision geometry), `body.height` (the
vertical extent of every link's collision geometry and sensor frames at zero joint position) and
`body.mass` are read lazily from the canonical URDF and are not overridable.

`RobotCaps.rendered_urdf()` is the canonical URDF of an instance ([`arena_robots.served_urdf`](../arena_robots/served_urdf.py)): the xacro of the robot or its assembly wrapper with frame overrides and default arguments, passive joints locked, assembly control joints injected and effort state interfaces ensured, mesh URIs untouched. `RobotCaps.served_urdf(...)` runs the same pipeline with the simulator's xacro arguments, sensors stripped per `optim`, unregistered `ros2_control` hardware stripped and sensor topics patched, and writes it with absolute `file://` mesh URIs to `<tempdir>/arena_robots/<sha256>.urdf`. Every simulator, MoveIt and the CHAMP bringup load that file. `RobotView.control_config(sim)`, `bringup_launch` and `srdf_source` are the other file accessors, and no module outside the arena_robots API reads robot files (`tests/test_robot_files_seam.py`).

`urdf_patches.yaml` in a robot directory or a component directory ([`arena_robots.urdf_patches`](../arena_robots/urdf_patches.py)) corrects the URDF of an instance right after the render, so `canonical()`, `serve()` and every fact derived from them see it. Each entry carries a `reason` and a `source` (where the corrected value comes from) and is one of:

- `link` plus `mass`: the link mass, its inertia scaled by the same factor.
- `total_mass`: every link the owner contributes scaled to the sum, masses and inertias. For a component these are the links its placement adds, recognized by the `<prefix><mount>_` stem, for a robot every other link.
- `joint` plus `limit` (`lower`, `upper`, `velocity`, `effort`).
- `link` plus `inertia_scale`.
- `link` plus `inertia: from_collision`: the inertia tensor of the link's single box, cylinder or sphere collision at the link's mass, rotated by the collision `rpy`, about an inertial origin set to the collision origin. A link without exactly one such collision raises.
- `link` plus `collision: {box: [x, y, z], xyz: [...], rpy: [...]}`: adds a box collision to the link (`xyz` and `rpy` default to zero), for geometry the vendor collision leaves out such as bumpers and fenders.

A family component (`arm/ur`) keys its entries under `variants:`, any other file lists them under `patches:`. Link and joint names are written as in the owner's own xacro, component names without the placement stem, and a name the render lacks raises.

A section of `caps.yaml` takes a `source: <url>` key, the document its values come from. It is read by `RobotCaps.source(<section>)` and `CapConfig.source`, kept out of the section data, the effective caps and their hash, and cannot be overridden.

`lint(view, parts, frames)` ([`arena_robots.urdf_lint`](../arena_robots/urdf_lint.py)) returns `Finding(check, severity, message)` rows on `served_urdf.canonical(...)`:

- `structure`: one root link, joint parents and children are links, the base frame is a link.
- `inertia`: mass above 0 on every link that moves, positive definite tensor, triangle inequality, trace within a factor 5 of a single primitive collision shape.
- `wheels`: every drive controller wheel joint is continuous with a cylinder or sphere collision.
- `joints`: revolute and prismatic joints carry a limit with effort above 0 and velocity above 0.
- `sensors`: every effective sensor frame is a link.

A finding is an error, a URDF fault is fixed in the URDF or by a patch. `tests/test_urdf_lint.py` lints every installed robot and fails on any error.

#### Where each value is authored

A value is written once, by its owner. The URDF (after `urdf_patches.yaml`) is the single source of everything it defines, `caps.yaml` holds everything else, each section with the `source` of its values where one exists. Every other file that carries a value is rendered from the effective caps, or `robots_check` warns where it differs.

| Value | Author it in | Read by |
| --- | --- | --- |
| Base velocity and acceleration the body can do (`mobile.limits`) | `mobile` section (`limits`, or `planar`) | flatsim drive, the drive controller keys of every axis `control*.yaml` flags (DiffDrive, Mecanum, swerve, omni), the CHAMP gait of `a1` and `go1`, the planner limits as their bound |
| Drive controller limit keys (`<axis>.min_velocity`, `max_velocity`, `min_acceleration`, `max_acceleration`) | `mobile.limits`, `control*.yaml` authors only the `has_velocity_limits` and `has_acceleration_limits` flags | written by `overlay_planar_limits` for every flagged axis in Gazebo, MuJoCo and Isaac, an infinite limit leaves the key unset |
| Wheel radius, track width, mecanum center projection, swerve and omni wheel positions, omni wheel angles | the URDF wheel joints and their round collision geometry, measured in the controller's `base_frame_id` | written over the drive controller's parameters by `effective_control_yaml` in every simulator, `RobotView.drive_geometry` and `urdf_facts.drive_geometry`, `control.yaml` keeps the joint names, multipliers and tuning |
| Drive controller frames (`base_frame_id`, `odom_frame_id`) | `base_frame` of `model_params.yaml` and `mobile.odom_frame` | written by `overlay_drive_frames` in every simulator, `RobotView.drive_controller`, never in `control*.yaml` |
| Flatsim pivot (`mobile.planar.pivot`) | `front` or `rear` in the `mobile` section, the number is the wheel joint x of the URDF | the flatsim drive model, provenance `urdf` |
| Continuous actions (`mobile.actions.continuous`) | `mobile.velocity_limits` unless authored | the RL action space, `characterization` |
| Controllers, command and odometry topics (`control.controllers`, `cmd_vel_topic`, `odom_topic`, `body.cmd_vel_topic`, `body.odom_topic`) | the control config: its `controller_manager` entries and the type of its drive controller | the simulators' controller spawn and odometry relay, the recorder |
| Sensors | the `sensor` blocks of the `assembly.yaml` components | `RobotView.effective_sensors` |
| Acoustic profile and joint torque clamp (`acoustics.*`, `max_joint_torque_nm`) | `config/acoustic_profile.yaml`, an `acoustics` section authors only the keys that differ | `AcousticsSpec`, the acoustics publisher, `RobotView.power_profile` for the power publisher |
| Power model wheels (`wheel_radius_m`, `num_wheels`) | the drive geometry and the drive controller's wheel joint count | `RobotView.power_profile`, the `power` section keeps the electrical and loss terms |
| CHAMP gait speeds | `mobile.limits` velocity bounds, the rest of the gait in `champ/gait.yaml` | the CHAMP bringup's `gait_config`, written per instance by `champ.served_champ_files` |
| CHAMP joint and link maps | the URDF: the four leg chains below the base frame, classified by their hip joint origins | the CHAMP bringup's `joints_map` and `links_map`, written per instance by `champ.served_champ_files` |
| Footprint and radius (`mobile.footprint`, `mobile.radius`) | the URDF collision geometry | the convex ground outline of the base (`body.outline`) and the circumradius of the effective footprint, provenance `urdf`, overridable per instance, never authored in the `mobile` section (`footprint_padding` stays authored) |
| Holonomy (`mobile.is_holonomic`) | the drive controller of `control.yaml` and `champ/gait.yaml` | true for a mecanum, swerve or omni drive controller or a CHAMP gait, false for diff drive or a base with `drive: false`, provenance `derived`, overridable per instance |
| Lift range (`lift.<name>.limits`), gripper effort (`gripper.<name>.max_effort`) | the lift or gripper joint's URDF `<limit>` | the direct adapters, authored only where a deliberate value differs from the URDF |
| Lidar geometry (`mobile.laser`) | the lidar component's xacro arguments (a bracket on the part) or the URDF | read from the scan sensor of the rendered URDF on the primary lidar (the first `laserscan` sensor), never authored in the `mobile` section |
| Arm chain links, chain joints and poses (`arm.<mount>.tip_link`, `chain`, `named_poses`, and `base_link` where the SRDF chain starts at the mount) | the SRDF group of the arm's `moveit.planning_group`: its `<chain>` links, its `group_state`s, and the non-fixed URDF joints between those links, provenance `derived`, `base_link`, other poses (`wave_*`) and the workspace stay authored | the arm adapters and task handlers, overridable per instance |
| Gripper open and close positions (`gripper.<name>.positions`) | the SRDF `open` and `close` group states of the gripper joint, provenance `derived` | the direct gripper adapter |
| Arm start pose (`initial_value` of a component's ros2_control joints) | the `stow` group state of the SRDF (`initial_value: stow` in the component) | every simulator |
| MoveIt joint limits (`arm.<mount>.joint_limits`) | the URDF `<limit>` for `max_velocity`, `min_position`, `max_position` and `max_effort` (a joint without a range gets `has_position_limits: false`), the component's `joint_limits.yaml` for `max_acceleration` and its flag, the scaling factors and any value it authors | `moveit_factory`, overridable per instance, tightened-only against the URDF `<limit>` |
| Joint position, velocity and effort limits | the URDF | the simulators, the bound of the MoveIt limits |
| Body mass | the URDF inertials, each component's in its xacro, corrected to the vendor's figure by `urdf_patches.yaml` (robot directory or component directory) | `body.mass`, the power model, the evaluation metrics through the caps snapshot |
| Collision geometry | the URDF | `body.outline`, `body.height` (the meshnav obstacle height with 0.1 m clearance) |
| Everything the URDF does not define (limits, speeds, polygons, actions, power and acoustics terms) | `caps.yaml`, with a `source` key in a section where a manufacturer document backs the values | the consumers of each cap |

#### `arm`, `lift`, `gripper`

The `moveit:` block declares what `moveit_factory.build_moveit_params` needs and assumes no
vendor. `planning_group` names the group the task handlers plan for; `srdf` and `joint_limits`
(each `{package, path}`) default into `package`; `args` are the xacro mappings of the served URDF and SRDF
(`prefix`/`mount`/`parent` also render the arm component's SRDF and joint limits per
placement); `robot_name` is the builder name (default: `package` minus `_moveit_config`, else the
robot). `package` is optional: a block with `srdf` and no `package` takes Arena-owned defaults
(OMPL with its stock planner configs). Every build, packaged or not, sets a KDL solver on the
planning group unless the package's kinematics file already names it, and maps the arm's
`controller` (FollowJointTrajectory over the `chain` joints) as the only MoveIt controller.
`tiago` declares its own `srdf/arm.srdf.xacro` and `srdf/joint_limits.yaml` this way; the `fr3`
and `ur` components do the same under `components/arm`, their `joint_limits.yaml` holding the acceleration only.
An SRDF composed from placement fragments also disables every joint's parent and child link
and every two links of one rigid body, both read off the rendered URDF.

The pipelines of a block without `package` are OMPL (the default) and Pilz, which `motion: linear`
reaches use. `moveit.cartesian_limits` sets the Pilz limits per arm over the defaults
`max_trans_vel: 0.5`, `max_trans_acc: 1.0`, `max_trans_dec: -1.0`, `max_rot_vel: 1.57`. Four
top-level arm keys tune reaches and scene sync, each read with a default:

| Key | Default | Meaning |
|---|---|---|
| `stall_error` | 0.2 | Tracking error in rad above which the arm counts as blocked |
| `stall_speed` | 0.05 | Joint speed in rad/s below which that joint counts as stopped, so an arm that lags while moving is not blocked |
| `stall_time` | 0.5 | Seconds the error must stay above `stall_error` on a stopped joint before the reach fails with `arm blocked` |
| `scene_radius` | 1.5 | Radius in m around the base within which episode objects enter the planning scene |

The SRDF named by `moveit.srdf` (rendered with `moveit.args`) is the source of the arm's
`tip_link`, `base_link` (when the arm authors none), `named_poses` (its `group_state`s of the planning group)
and of a gripper's `positions` (the `open` and `close` states of its joint). `chain` is the non-fixed
URDF joints from the SRDF chain's base to its tip. `controller` and the adapter sub-blocks stay authored.
Loader precedence per field: authored > SRDF and URDF derived > raise. A component whose arm mounts on a
parent frame other than its SRDF chain base (`arm/ur`, `arm/fr3`) authors `base_link: ${prefix}${parent}`.

`positions` holds the joint values of `open` and `close`, `max_effort` the effort every gripper
command carries (the joint's URDF effort unless authored). Gripper components carry the same block under `caps:`.

#### Adapter sub-blocks are namespaced, not dispatch

`moveit:`, `nav2:`, `rl:`, `drl_grasp:`, ... inside a section are pure data
organization. **No in-file handler declaration**: adapters are selected at
runtime (CLI / setup YAML) and are handed the section, reading whichever
primitives + adapter-sub-block they know about.

### `control.yaml`

ROS 2 `controller_manager` configuration. Define `joint_state_broadcaster`
plus whichever controller drives the robot (e.g.
`diff_drive_controller/DiffDriveController` for wheeled, `JointTrajectoryController`
for an arm). See [`husky/control.yaml`](husky/control.yaml) for a diff-drive
example and [`tiago/control.yaml`](tiago/control.yaml) for an
arm-extended example.

An optional sibling `control.<sim>.yaml` (`sim` a `SimSimulator` value such as
`gazebo`, `isaac` or `mujoco`) overlays `control.yaml` key by key in that
simulator only, for a value the simulator's physics needs and the real robot
does not. [`jackal/control.gazebo.yaml`](jackal/control.gazebo.yaml) sets the
`wheel_separation_multiplier` Gazebo's skid steer turns at.

The drive controller's limit values are not written in `control.yaml` either, only its `has_velocity_limits` and `has_acceleration_limits` flags per axis, the keys come from `mobile.limits` at spawn. The drive controller's `base_frame_id` and `odom_frame_id` are not written in `control.yaml` either, `effective_control_yaml` sets them from `base_frame` and `mobile.odom_frame` of the instance. The drive controller's geometry is not written in `control.yaml`: `wheel_radius`,
`wheel_separation` (diff drive), `kinematics.wheels_radius` and
`kinematics.sum_of_robot_center_projection_on_X_Y_axis` (mecanum),
`wheel_positions_x`, `wheel_positions_y` (swerve at the steering joints, omni at
the wheel joints) and `wheel_angles` (omni, the rolling direction of each wheel
axle) come from the URDF at spawn.

Every simulator spawns a joint at the `initial_value` of its ros2_control
position state interface, so a legged robot's `urdf/base_hw/<name>.ros2_control.urdf`
carries its standing pose there (see [`go1`](go1/urdf/base_hw/go1.ros2_control.urdf)).

### Gazebo bridge rows

A robot has no bridge file. Gazebo's per-robot `parameter_bridge` takes its rows from the
canonical URDF: `Sensor.plugin_bridge_rows` maps the model's `gz-sim-pose-publisher-system`
plugin (`publish_model_pose` true) to `/model/<name>/pose` as `pose`, and its
`gz-sim-odometry-publisher-system` and `gz-sim-diff-drive-system` plugins without an
`<odom_topic>` to `/model/<name>/odometry` as `odom`. `Sensor.bridge_rows` maps every sensor
output that `effective_sensors` declares, so the `sensors:` entries and the URDF are the only
sources.

### `assembly.yaml`: component-catalog robots only

Sibling of `model_params.yaml` (deliberately not inside it: the existing scalar
`priority:` field would collide). Its presence declares the robot's sensors are sourced
from the component catalog: sensors come from `components/` via mounts, and the
chassis xacro carries no sensor invocations.

```yaml
prefix: ""            # frame-prefix convention; omit for the Robotnik "robot_" default
mounts:
  front_laser:        # mount names adopt the chassis xacro's own frame names
    parent: chassis_link
    xyz: [0.53, 0.33, 0.1145]     # verbatim from the chassis xacro's sensor invocation
    rpy: [3.141592653589793, 0, 0.7853981633974483]
    accepts: [lidar]              # a set: a mount may accept more than one part type
defaults:
  lidar:
    - variant: sick_s300
      mount: front_laser
      overrides: {name: lidar_rear, topic: scan/rear}   # per-instance tuning lives HERE
```

`accepts` is a set, not a scalar: a versatile mounting point (e.g. a top plate) may
accept several part types. When more than one mount accepts the same type, allocation
preference among them follows mount DECLARATION ORDER in this file, first-declared
wins. There is no separate `priority:` field; reorder the `mounts:` block itself to
change preference.

Requests (`robot:=name[lidar=x,...]`) resolve against this file with replace-on-touch
semantics; see the grammar section in
`arena_runtime/arena_runtime/arena_runtime/env/README.md`.

A sensor that is physically present but deliberately left unbridged (e.g. a GPS/navsat
mast) stays inline in the robot's own xacro; it gets no mount or component entry here.

### Moving a robot's sensors to the component catalog

1. **Golden capture first**: render the robot's current URDF via xacro in-container and
   save to `tests/golden/<name>_default.urdf` (strip the container wrapper's non-XML
   stdout prefix; verify it parses).
2. Author/reuse `components/` entries (see [components/README.md](../components/README.md)).
3. Write `assembly.yaml` with mount origins copied verbatim from the xacro invocations
   (verify each sensor's actual parent link; robots differ).
4. Strip ONLY the sensor invocations + now-unused includes from the chassis xacro,
   leaving no comment behind; never delete macro files, other robots may include
   them. Ensure
   the wrapper-contract args exist (`namespace`, `prefix`, `gazebo_classic`,
   `gazebo_ignition`) with defaults preserving the chassis's existing values.
5. `tests/test_default_morphology.py` parametrizes over every `tests/golden/*_default.urdf`
   automatically, so the robot is now covered: effective-sensors consistency
   (`effective_sensors({})` has unique names under the namespace) and canonicalized URDF
   structural comparison against the golden from step 1.
6. The golden reproduces known drift byte-identically; this move is not a bug-fix
   opportunity. Dead (env-gated Gazebo-Classic) blocks stay untouched.

### `preset.yaml`: a robot that is a base robot with parts

A directory holding only `preset.yaml` is a preset, a robot name that stands for another robot with fixed parts and cap overrides:

```yaml
robot: rbvogui
parts: {arm: ur5e}
caps: {}
```

`robot:=rbvogui_plus`, `rbvogui_plus[...]`, setup file and fleet entries and scenario `robots:` entries expand in `SetupFile.Config.parse` to the base robot with the preset's `parts` (the directives of `assembly.yaml` requests) and `caps` (`<cap>.<path>` overrides) merged under the entry's own brackets, which win wherever both touch the same socket, type or cap path. The base robot and every part must exist and place, or loading the preset raises. `robots_catalog`, `RobotIdentifier.listall` and `arena robot` list a preset as a robot with its base and parts, and its submodules are those of the base and of the components its parts place.

## Optional files

| Path | Purpose |
| --- | --- |
| `urdf/<name>.urdf.xacro` | robot description; mesh refs use `package://arena_robots/robots/<name>/meshes/…` or a fixed upstream package (`package://jackal_description/…`) |
| `meshes/` | STL/DAE/OBJ files, normally a git submodule under `github.com/arena-robots/<name>` (opt-in via `arena feature robots add <name>`) |
| `launch/` | robot-specific launch files (e.g. nav2 overrides) |
| `launch/bringup.launch.py` | per-robot sidecar (a legged robot's gait controller, `cmd_vel` to joint trajectories), included by `RobotManager` once per spawn inside the robot namespace in every simulator, gated only on the file existing, with launch arguments `base_frame`, `use_sim_time` and the required `urdf_path`, the served URDF of the instance (the CHAMP bringups of `a1` and `go1` read their robot description from it and also require `joints_map`, `links_map` and `gait_config`) |
| `README.md` | free-form robot docs (upstream references, env vars, etc.) |

## Meshes

Every robot that needs per-robot geometry has a `meshes/` git submodule pinned
via `.gitmodules`, pointing at `github.com/arena-robots/<name>.git` with
`update = none`. Running `arena feature robots add <name>` clones it; config
edits you make under the robot dir stay in the main Arena repo. A robot whose
upstream repo is not a buildable ROS 2 package (p3at, tiago) instead points its
submodules under `meshes/` at that upstream repo with `sparse =` on the mesh
directory, and its URDF references `package://arena_robots/robots/<name>/meshes/<path in the upstream repo>`.
Robots that use upstream geometry (jackal, turtlebot) have no `meshes/` submodule,
their URDFs reference `package://jackal_description/…` etc., supplied by
`deps/jackal` or `deps/turtlebot4`.

## Checklist

1. `mkdir -p arena_robots/arena_robots/robots/<name>/caps`
2. Write `model_params.yaml` (minimal identity), `control.yaml`.
3. Write `caps.yaml` with a `mobile` section if the robot has a base (`drive: false` for one that
   only seats it, such as `fr3`), a robot without one is immobile. Add `arm` and/or `lift` sections
   if the robot has those subsystems.
4. (Optional) add `urdf/<name>.urdf.xacro` and/or a `meshes/` submodule.
   For component-catalog sensors, add `assembly.yaml` + a golden (see "Moving a
   robot's sensors to the component catalog" above).
5. If the robot ships an upstream ROS package, add it as a submodule with
   `robot = <name>` (and `update = none`) in `.gitmodules`.
6. `arena feature robots add <name>` to fetch any submodules.
7. `arena feature robots check` to verify every `package://arena_robots/…`
   URI resolves on disk.
8. `arena launch … robot:=<name>` to bring it up.

## Adding a new cap

When a new subsystem kind (e.g. `gripper`, `ptu_head`) becomes needed:

1. Add an entry to the Active cap vocabulary table above with its file, shape,
   meaning, and list of robots that have it.
2. Define the primitive fields (what every adapter-agnostic consumer can count
   on) and document adapter sub-block conventions if any exist.
3. If adding a typed accessor is warranted, extend
   [`caps.py`](../arena_robots/caps.py) with a `<Cap>Spec` subclass. The loader dict-keyed
   pattern is uniform, see `ArmSpec`/`LiftSpec` for the template.
