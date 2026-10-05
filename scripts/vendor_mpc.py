"""Copy the modules of shockbench_flow that the MPC planners need into an agent folder as the package ``sbfv``.

The scoring container guarantees only numpy, SciPy and torch to participant code, so the planner ships with the
agent. Imports are rewritten from ``shockbench_flow`` to ``sbfv`` so the copy never mixes with the kit's own package;
highspy becomes SciPy's bundled HiGHS core, fastjsonschema and loguru are dropped (the instance comes from Static,
already validated by the server), and the keyed draws take NumPy's reference path instead of the numba kernels.
"""

import ast
import re
import shutil
import sys
from pathlib import Path

import fire
import shockbench_flow

SEEDS = (
    "shockbench_flow.policies.mpc_det",
    "shockbench_flow.policies.mpc_scen",
    "shockbench_flow.policies.lp_common",
    "shockbench_flow.instance.io",
    "shockbench_flow.disruption.profiles",
)
ALLOWED = set(sys.stdlib_module_names) | {"numpy", "scipy", "torch", "__future__", "shockbench_flow"}

HIGHSPY_SHIM = '''def _highspy():
    """SciPy's bundled HiGHS core under highspy's names (the policy image has SciPy, not highspy)."""
    import types

    from scipy.optimize._highspy import _core

    ns = types.SimpleNamespace(**{n: getattr(_core, n) for n in dir(_core) if not n.startswith("__")})
    ns.Highs = _core._Highs
    return ns
'''

PATCHES = {
    "__init__.py": [
        (re.compile(r"\ntry:\n    from loguru.*?del _logger\n", re.S), "\n"),
    ],
    "policies/lp_common.py": [
        (re.compile(r"if TYPE_CHECKING:.*?\n    import highspy\n\n"), "if TYPE_CHECKING:\n"),
        (re.compile(r'def _highspy\(\):\n    """.*?"""\n    import highspy\n\n    return highspy\n', re.S), HIGHSPY_SHIM),
    ],
    "parallel.py": [
        (
            re.compile(r"    from joblib import Parallel, delayed\n\n    return list\(Parallel\(.*?\)\n"),
            "    return [fn(x) for x in xs]\n",
        ),
    ],
    "omega/seeds.py": [
        (re.compile(r'^KERNELS = os\.environ\.get\(KERNELS_VAR, "1"\) != "0"$', re.M), "KERNELS = False  # no numba"),
    ],
    "instance/io.py": [
        (re.compile(r"^import fastjsonschema\n", re.M), ""),
        (re.compile(r"(def _validate_schema\(raw: Any\) -> None:\n    \"\"\".*?\"\"\"\n)", re.S), r"\1    return\n"),
        (re.compile(r"err: fastjsonschema\.JsonSchemaValueException"), "err"),
        (re.compile(r"return fastjsonschema\.compile\(.*?\)\n"), "raise NotImplementedError\n"),
        (re.compile(r"except fastjsonschema\.JsonSchemaValueException as err:"), "except ValueError as err:"),
    ],
}


def module_file(root: Path, name: str) -> Path | None:
    parts = name.split(".")[1:]
    if not parts:
        return root / "__init__.py"
    rel = Path(*parts)
    for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def top_level_imports(tree: ast.Module):
    """Import statements executed when the module loads: not inside a function, not under TYPE_CHECKING."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            if "TYPE_CHECKING" not in ast.unparse(node.test):
                stack.extend(node.body + node.orelse)
        elif isinstance(node, ast.Try):
            stack.extend(node.body + node.orelse + node.finalbody + [s for h in node.handlers for s in h.body])
        elif isinstance(node, (ast.With,)):
            stack.extend(node.body)


def imports_of(path: Path, name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    found = set()
    for node in top_level_imports(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            if node.level:
                base = package.split(".")
                base = base[: len(base) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module
            names = [mod] + [f"{mod}.{a.name}" for a in node.names]
        found.update(n for n in names if n.startswith("shockbench_flow"))
    return found


def closure(root: Path) -> dict[str, Path]:
    todo, files = list(SEEDS), {}
    while todo:
        name = todo.pop()
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub in files:
                continue
            path = module_file(root, sub)
            if path is None:
                continue
            files[sub] = path
            todo.extend(imports_of(path, sub))
    return files


def foreign_imports(path: Path) -> set[str]:
    """Every import in the file, at any depth, outside the standard library, numpy, SciPy, torch and sbfv."""
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.add(node.module.split(".")[0])
    return out - ALLOWED - {"sbfv"}


def main(dest="agents/mpc/sbfv"):
    root = Path(shockbench_flow.__file__).parent
    files = closure(root)
    out = Path(dest)
    if out.exists():
        shutil.rmtree(out)
    for _name, path in sorted(files.items()):
        rel = path.relative_to(root).as_posix()
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\bshockbench_flow\b(?=\.|\s+import|\s*$)", "sbfv", text, flags=re.M)
        for pattern, repl in PATCHES.get(rel, []):
            text, n = pattern.subn(repl, text)
            if n == 0:
                raise RuntimeError(f"patch {pattern.pattern[:40]!r} did not apply to {rel}")
        target.write_text(text, encoding="utf-8")
    for pkg in {p.parent for p in out.rglob("*.py")}:
        (pkg / "__init__.py").touch()
    print(f"{len(files)} modules -> {out}")
    for path in sorted(out.rglob("*.py")):
        bad = foreign_imports(path)
        if bad:
            print(f"  {path.relative_to(out)}: still imports {sorted(bad)}")


if __name__ == "__main__":
    fire.Fire(main)
