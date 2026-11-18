# Robot setup files

A robot setup file (`config/setup/<name>.yaml`) is a YAML list declaring which
robot instances to spawn for a given scenario. Resolved at runtime by
[`RobotSetupIdentifier`](../../arena_robots/SetupFile.py): `RobotSetupIdentifier('demo')`
loads `demo.yaml` and returns a flat `list[Config]` (one per instance, with
`count` already expanded).

## Schema

Each list entry is either a **string** (shorthand) or a **dict** (full form).
Both resolve into one or more [`Config`](../../arena_robots/SetupFile.py) objects.

### Shorthand: one instance of the named robot

```yaml
- jackal
- turtlebot
```

Equivalent to `{ robot: jackal, name: jackal }` followed by
`{ robot: turtlebot, name: turtlebot }`.

### Dict form: full control

Parsed by [`Config.parse`](../../arena_robots/SetupFile.py). Keys route by
grammar, not a fixed field list:

```yaml
- robot: jackal            # required: directory name under robots/<robot>/
  name: jackal_lead        # optional: instance name / namespace prefix (default: unset, see below)
  count: 3                 # optional: expand into N independent Configs (default: 1)
  pos: [1.0, 2.0, 0.0]     # optional: initial pose [x, y, yaw], consumed by Robot.parse (default: (0, 0, 0))
  adapters:                # optional: cap -> adapter kind
    mobile: nav2
    arm: moveit
  mobile.adapter: nav2     # optional: dotted shorthand, equivalent to the adapters: block above
  frames:                  # optional: mount -> frame stem override, validated against the robot's assembly mounts
    top: custom_laser_link
  frames.front: custom_front_link  # optional: dotted shorthand, one mount per key
  mobile.velocity_limits.linear: [-0.5, 1.0]  # optional: cap override, see below
  mobile.limits.velocity: {scale: 0.5}         # optional: cap override scaling every number below the path
  top_tool: gripper/franka_hand[max_effort=*0.5]  # optional: a part's own cap overrides in brackets after its variant
  extra:                   # optional: dict merged verbatim into the parsed robot's extra data
    foo: bar
  top: lidar/sick          # any other bare key is a morphology directive (mount=type/variant, mount=variant, type=variant, or lhs=none), requires the robot to have an assembly.yaml
```

`count` is consumed during parsing and never appears on the resulting
`Config`: set `count: 3` and you get three independent instances (each with
its own deep-copied `extra`/`parts`/`frames`/`caps`), all sharing the same `name`
field. If you need distinct names per instance, emit multiple list entries
instead of using `count`.

`name` omitted (dict form) leaves the instance anonymous: robots_manager
groups anonymous instances by robot model and assigns `<robot>` or, when more
than one anonymous instance of the same robot exists, `<robot>_0`,
`<robot>_1`, etc. The shorthand string form (`- jackal`) always sets an
explicit `name` equal to the robot name, so it is never anonymous.

`robot`, `name`, `count`, `parts`, `frames`, `adapters`, `caps`, `extra` are reserved
field names and cannot be reused as morphology directive keys. A top-level
`parts:` dict is rejected for the same reason: directives must be given as
individual bare keys. Dotted keys are `<cap>.adapter: <kind>`, `frames.<mount>: <stem>`
and cap overrides `<cap>.<path>: <value>`. `adapters` values are
validated against the registered adapter kinds per cap (`mobile`:
`nav2`, `external`, `manual`, `rosnav_rl`, `drl`, `none`, `test-collision`,
`arm`: `moveit`, `none`, `lift`: `direct`, `none`, `gripper`: `direct`, `none`, see
[env/adapters](../../../../arena_runtime/arena_runtime/arena_runtime/env/adapters/README.md)).
Morphology directives and `frames` are only valid for a robot whose
`assembly.yaml` defines mounts. Both raise if the model has none.

### Cap overrides

A dotted key `<cap>.<path>` overrides values of the instance's
`caps/<cap>.yaml`, for this instance only. A path segment is a key, an integer
list index (`mobile.footprint.0`, negative counts from the end) or `*` for every
key or element at that level (`mobile.velocity_limits.*`, `arm.*.stall_time`).
Every parent of the path must exist in the cap, except a top-level field the cap
leaves unset (`mobile.acceleration_limits.linear: 2.0`) or an adapter block
such as `mobile.meshnav.planner: cvp`. The value is one of:

| Value | Inline | Effect |
| --- | --- | --- |
| any YAML value | `0.5`, `[-0.5,1.0]`, `smac` | replaces what is at the path, or adds the key to an existing mapping |
| `{scale: a, offset: b}` (either key) | `*a`, `+b`, `*a+b`, `*a-b` | `a*v + b` on every number below the path (a scalar, both ends of a range, every axis of a block) |
| `{append: [...]}` | `+[<item>,...]` | extends a list |
| `{delete: true}` | `!delete` | removes a key or list element |

Overrides apply in order, so a named instance after a `*` refines it.

A part placed through `assembly.yaml` takes its own overrides in brackets
after its variant, paths within its cap instance: `top: arm/fr3[stall_time=2.0]`,
`gripper: franka_hand[max_effort=*0.5,positions.open=0.04]`. They follow the
part to whatever mount it resolves to, so a type directive without a mount
works too. A `<cap>.<mount>.<path>` key aimed at a placed part raises and names
the bracket form. Robots whose `arm`, `lift` or `gripper` instances come from
their static `caps/<cap>.yaml` (no assembly) address them as
`<cap>.<instance>.<path>`, the instance being the key in that file.

Overrides are validated when the robot is parsed: an unknown cap, instance or
field, an index out of range, a `*` matching nothing, an operation on an unset or
non-numeric value, or a value its typed field rejects raises with the robot and
the path.

The same keys go inline in `robot:=`:
`robot:=jackal[top=arm/fr3[stall_time=2.0],top_tool=gripper/franka_hand[max_effort=*0.5],mobile.limits.velocity.linear=[-0.5,0.5],mobile.inflation_radius=+0.1]`.
A mapping value has no inline form, override its fields one by one.
`arena robot paths '<entry>'` and the MCP tool `robots_overrides` list every path
an instance accepts with its effective value, and TAB after `robot:=` in `arena
launch` completes them.

`mobile.limits` (`velocity` and `acceleration` per axis `linear`, `lateral`,
`angular`) is what the base can physically do. A robot without it in its file
takes it from `planar.velocity` and `planar.acceleration`, and the effective cap
carries it in their place: flatsim drives by it, and an instance that overrides
`limits` or `planar` gets the base controller's velocity and acceleration limits
in `control.yaml` replaced from it in Gazebo, MuJoCo and Isaac.
`velocity_limits` and `acceleration_limits` are what planners, the DRL clamp and
the training action space use, clamped to `limits` on every axis the file or an
override gives `limits` explicitly. `mobile.limits.velocity.linear=[-0.5,0.5]`
or `mobile.limits=*0.5` slows the robot and its planners together, a planner
limit set narrower stays narrower.

Changing an instance's overrides respawns and relaunches it. The env-wide
`robot.<cap>.<key>:=` launch arguments only reach the bound adapter's own
sub-block, a top-level cap field there raises.

Nav2 planner/controller/behavior-tree tuning is not a robot setup file key:
it is selected at launch time (`robot.planner:=`, see
[arena_bringup/launch/README.md](../../../../arena_bringup/launch/README.md)).

## Examples shipped with arena_robots

- [`all_robots.yaml`](all_robots.yaml): one of each known robot using
  shorthand; useful for smoke-testing every URDF and mapping file.
- [`demo.yaml`](demo.yaml): three jackals via `count: 3`.
