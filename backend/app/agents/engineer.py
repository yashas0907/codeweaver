"""Deterministic code transformations the agent can apply without an LLM.

These are real, verifiable code surgeries:
  - paginate_function: adds limit/offset parameters to a list-returning
    function and slices its result (AST-guided, formatting-preserving
    outside the edited region).
  - repair_import: fixes a broken import by locating the real module/symbol
    in the repository graph.

Each function returns (old_text, new_text) replacement pairs that the caller
applies through the workspace sandbox, so every change is diffed and
reviewable. They refuse to act when the pattern is not unambiguously
present — no guessing.
"""

from __future__ import annotations

import ast
import re


class TransformError(RuntimeError):
    """The transformation cannot be applied safely."""


def _walk_func_body(node: ast.stmt) -> list[ast.stmt]:
    """Top-level statements inside a function/class body (stop at nested defs/classes)."""
    stmts: list[ast.stmt] = []
    stack = [node]
    while stack:
        current = stack.pop()
        for child in reversed(list(ast.iter_child_nodes(current))):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.stmt):
                stmts.append(child)
                stack.append(child)
    return stmts


def _collect_list_targets(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict]:
    """Return list-targets for a function whose body we've already extracted."""
    body_stmts = _walk_func_body(func)
    for stmt in body_stmts:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            value = stmt.value
            if isinstance(value, (ast.Name, ast.List, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.Tuple)):
                return [{
                    "name": func.name, "line": func.lineno,
                    "end_line": func.end_lineno or func.lineno,
                    "return_expr": ast.unparse(value), "return_line": stmt.lineno,
                }]
    return []


def find_list_returning_functions(source: str) -> list[dict]:
    """Top-level functions AND class methods that return collections."""
    targets: list[dict] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return targets
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            targets.extend(_collect_list_targets(node))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    targets.extend(_collect_list_targets(child))
    return targets


def paginate_function(source: str, func_name: str, default_limit: int = 20) -> tuple[str, str]:
    """Return (old_text, new_text) for the function's return line.

    Works for both top-level functions and class methods.
    Uses direct text manipulation — not AST unparse — so formatting is preserved.
    """
    lines = source.splitlines()

    # Find the def line (top-level or indented inside a class)
    func_start = -1
    for i, ln in enumerate(lines):
        stripped = ln.lstrip()
        if (stripped.startswith(f"def {func_name}") or
            stripped.startswith(f"async def {func_name}")):
            func_start = i
            break
    if func_start < 0:
        raise TransformError(f"function {func_name!r} not found")

    base_indent = len(lines[func_start]) - len(lines[func_start].lstrip())

    # Find where the function ends: first non-blank line at/under base_indent (after body)
    func_end = len(lines)
    for i in range(func_start + 1, len(lines)):
        ln = lines[i]
        stripped = ln.strip()
        # Stop at first non-blank line that dedents to base_indent or less
        if stripped and indent_of(ln) <= base_indent:
            func_end = i
            break

    func_lines = lines[func_start:func_end]

    # Find the return to paginate.
    # Strategy: scan all lines. At indent == base_indent+4 (function-body level),
    # skip compound keywords (if/try/with/for/while) and keep looking for a
    # return at that level. Also accept the FIRST return at deeper indent (inside
    # a compound block) as a fallback.
    return_idx = -1
    seen_deep_return = False
    for i, ln in enumerate(func_lines[1:], start=1):
        stripped = ln.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = indent_of(ln)

        if indent <= base_indent:
            break

        if re.match(r"^\s*(def|class|async)\s+", stripped):
            break

        if indent == base_indent + 4:
            if re.match(r"^\s*(if|elif|else|try|except|finally|with|for|while)\b", stripped):
                continue
            m = re.match(r"^\s*return\s+(.+)", stripped)
            if m and m.group(1).strip() not in ("None",):
                return_idx = i
                break
        elif return_idx < 0 and seen_deep_return is False:
            # Deep return (inside a compound block): remember it but keep looking
            # for a shallower one
            m = re.match(r"^\s*return\s+(.+)", stripped)
            if m and m.group(1).strip() not in ("None",):
                return_idx = i
                seen_deep_return = True

    if return_idx < 0:
        raise TransformError(f"function {func_name!r} has no direct return to paginate")

    old_return = func_lines[return_idx]
    rest = old_return.strip()[6:].strip()
    new_return = old_return[:len(old_return) - len(old_return.lstrip())] + f"return {rest}[offset:offset + limit]"

    new_lines = list(func_lines)
    new_lines[return_idx] = new_return

    # Add limit/offset params to def line if absent
    old_def = func_lines[0]
    if "limit" not in old_def and "offset" not in old_def:
        if "()" in old_def:
            # No existing params: replace () with (limit, offset)
            new_def = old_def.replace("()", f"(limit: int = {default_limit}, offset: int = 0)", 1)
        else:
            # Has existing params: add ,limit, offset before closing )
            new_def = old_def.replace(")", f", limit: int = {default_limit}, offset: int = 0)", 1)
        new_lines[0] = new_def

    return "\n".join(func_lines), "\n".join(new_lines)


def indent_of(ln: str) -> int:
    return len(ln) - len(ln.lstrip())


def _paginate_ast(func: ast.FunctionDef | ast.AsyncFunctionDef, default_limit: int):
    # 1) add parameters with defaults: appended args simply extend the
    #    defaults array (it aligns to the trailing positional args).
    limit_arg = ast.arg(arg="limit", annotation=ast.Name(id="int", ctx=ast.Load()))
    offset_arg = ast.arg(arg="offset", annotation=ast.Name(id="int", ctx=ast.Load()))
    func.args.args = list(func.args.args) + [limit_arg, offset_arg]
    func.args.defaults = list(func.args.defaults) + [
        ast.Constant(value=default_limit), ast.Constant(value=0),
    ]

    # 2) wrap the first collection return with slicing
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            if isinstance(value, (ast.Name, ast.List, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.Tuple)):
                node.value = ast.Subscript(
                    value=value,
                    slice=ast.Slice(
                        lower=ast.Name(id="offset", ctx=ast.Load()),
                        upper=ast.BinOp(
                            left=ast.Name(id="offset", ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Name(id="limit", ctx=ast.Load()),
                        ),
                        step=None,
                    ),
                    ctx=ast.Load(),
                )
                # Slicing a generator expression is invalid; materialize it.
                if isinstance(value, (ast.GeneratorExp,)):
                    node.value.value = ast.ListComp(elt=value.elt, generators=value.generators)
                break
    ast.fix_missing_locations(func)
    return func


# --------------------------------------------------------------------------- #
# Import repair
# --------------------------------------------------------------------------- #

IMPORT_LINE = re.compile(r"^(\s*)(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.,\s]+))$", re.MULTILINE)


def repair_import(
    source: str,
    missing_name: str,
    source_module_hint: str,
    symbol_locations: dict[str, str],
) -> tuple[str, str] | None:
    """Fix `from X import missing_name` when the symbol actually lives elsewhere.

    symbol_locations maps symbol/module names -> repository file paths
    (module style: `app/auth/service`). Returns (old_text, new_text) or None.
    """
    target_line = None
    for m in IMPORT_LINE.finditer(source):
        from_mod, imported = m.group(2), m.group(3)
        if from_mod is None:
            continue
        names = [n.strip().split(" as ")[0] for n in imported.split(",")]
        if missing_name in names:
            target_line = (m.group(0), from_mod)
            break
    if target_line is None:
        return None

    old_text, from_mod = target_line
    # candidate modules that may define the symbol
    for sym_key, path in symbol_locations.items():
        if sym_key == missing_name or sym_key.endswith(f"/{missing_name}") or sym_key.endswith(f".{missing_name}"):
            new_mod = path.replace("/", ".").removesuffix(".py")
            if new_mod == from_mod:
                return None
            new_text = old_text.replace(f"from {from_mod} import", f"from {new_mod} import")
            return old_text, new_text
    return None


def fix_module_import_path(
    source: str, missing_module: str, module_index: dict[str, str]
) -> tuple[str, str] | None:
    """Fix `import missing.module` / `from missing.module import ...` when the
    module exists under a different path in the repository."""
    candidates = []
    short = missing_module.split(".")[-1]
    for mod, path in module_index.items():
        if mod == missing_module or mod == short or mod.endswith(f".{short}"):
            candidates.append((mod, path))
    if not candidates:
        return None
    new_module = candidates[0][0]
    patterns = [
        (f"from {missing_module} import", f"from {new_module} import"),
        (f"import {missing_module}", f"import {new_module}"),
    ]
    for old, new in patterns:
        if old in source:
            return old, source.replace(old, new, 1)
    return None


def rewrite_import_to_symbols(
    source: str, stale_module: str, symbols: dict[str, str]
) -> tuple[str, str] | None:
    """Repair `from stale_module import A, B` when A/B now live elsewhere.

    `symbols` maps symbol name -> repository file path. The rewrite targets
    the module where the most imported symbols are currently defined. Only
    fires when at least one imported name is found — no blind guessing.
    """
    if not stale_module or not symbols:
        return None
    for m in IMPORT_LINE.finditer(source):
        from_mod, imported = m.group(2), m.group(3)
        if not from_mod:
            continue
        if from_mod != stale_module and not from_mod.endswith("." + stale_module):
            continue
        names = [n.strip().split(" as ")[0] for n in imported.split(",") if n.strip()]
        mod_votes: dict[str, int] = {}
        for name in names:
            path = symbols.get(name)
            if not path:
                continue
            mod = path.replace("/", ".").removesuffix(".py").removesuffix(".__init__")
            mod_votes[mod] = mod_votes.get(mod, 0) + 1
        if not mod_votes:
            continue
        best_mod = max(mod_votes.items(), key=lambda kv: kv[1])[0]
        if best_mod == from_mod:
            continue
        old = m.group(0)
        new = old.replace(f"from {from_mod} import", f"from {best_mod} import")
        return old, new
    return None