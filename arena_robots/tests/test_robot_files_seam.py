from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
API = "arena_robots/arena_robots/arena_robots"
API_MODULES = frozenset(f"{API}/{name}.py" for name in ("Robot", "caps", "urdf_facts", "served_urdf", "urdf_patches", "urdf_lint", "catalog", "drift", "assembly"))
ALLOWLIST = {"_meta/arena_cli/features/robots.py": "runs on the host without ROS, where arena_robots cannot be imported"}
SKIPPED_DIRS = frozenset({".git", "build", "install", "log", "node_modules", "__pycache__", "tests", "deps", ".venv", ".uv-workspace", "repos"})
ROBOT_FILES = frozenset({"model_params.yaml", "control.yaml", "mappings.yaml", "assembly.yaml", "component.yaml"})
ROBOT_DIRS = frozenset({"urdf", "srdf", "caps", "components"})
XACRO_SUFFIXES = (".urdf.xacro", ".srdf.xacro")
PLAIN_LOOKUPS = frozenset({"get", "pop", "setdefault", "startswith", "endswith", "replace", "format"})
SHARED_CONFIG_CALLS = frozenset({"nav2_cfg"})


def _sources() -> Iterator[Path]:
    stack = [REPO]
    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                if entry.name not in SKIPPED_DIRS:
                    stack.append(entry)
            elif entry.suffix == ".py" and not entry.name.startswith("test_"):
                yield entry


def _is_robot_file(value: str) -> bool:
    return value in ROBOT_FILES or any(value.endswith(f"/{name}") for name in ROBOT_FILES)


def _call_name(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _strings(node: ast.AST) -> Iterator[ast.Constant]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child


def _docstrings(tree: ast.Module) -> set[int]:
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = _docstrings(tree)
    found: list[str] = []

    def flag(node: ast.AST, what: str) -> None:
        found.append(f"{path.relative_to(REPO)}:{node.lineno}: {what}")  # pyright: ignore[reportAttributeAccessIssue]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] == "xacro" for alias in node.names):
            flag(node, "imports xacro")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "xacro":
            flag(node, "imports xacro")
        elif isinstance(node, (ast.Name, ast.Attribute)) and (node.id if isinstance(node, ast.Name) else node.attr) in {"process_file", "render_wrapper_xacro"}:
            flag(node, "renders a xacro")
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            value = node.right.value
            if value in ROBOT_DIRS or _is_robot_file(value) or value.endswith(XACRO_SUFFIXES):
                flag(node, f"joins a robot path {value!r}")
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str) and part.value.endswith(XACRO_SUFFIXES):
                    flag(node, "builds a xacro file name")
        elif isinstance(node, ast.Call) and _call_name(node) not in PLAIN_LOOKUPS | SHARED_CONFIG_CALLS:
            for arg in [*node.args, *(keyword.value for keyword in node.keywords)]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and id(arg) not in docstrings:
                    value = arg.value
                    if _is_robot_file(value) or value.endswith(XACRO_SUFFIXES):
                        flag(node, f"passes a robot file {value!r}")
                    elif value == "xacro" or value.startswith("xacro "):
                        flag(node, "runs the xacro command")
        elif isinstance(node, (ast.List, ast.Tuple)):
            for element in node.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str) and (element.value == "xacro" or element.value.startswith("xacro ")):
                    flag(node, "runs the xacro command")
    return found


def test_robot_and_component_files_are_read_only_through_the_arena_robots_api():
    violations: list[str] = []
    for path in _sources():
        relative = path.relative_to(REPO).as_posix()
        if relative in API_MODULES or relative in ALLOWLIST:
            continue
        violations.extend(_violations(path))
    assert not violations, "robot or component files read, or xacro rendered, outside the arena_robots API:\n" + "\n".join(violations)


def test_the_allowlist_names_existing_files_with_a_reason():
    for relative, reason in ALLOWLIST.items():
        assert (REPO / relative).is_file(), relative
        assert reason
