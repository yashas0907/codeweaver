"""AST-based Python file analysis.

Extracts symbols, imports, calls and HTTP routes from a Python source file
using the `ast` module — never regex — so line numbers and structure are
exact. Files with syntax errors still produce partial facts plus the error.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass
class ParsedSymbol:
    name: str
    kind: str                     # function | class | method | test | route
    line_start: int
    line_end: int
    parent: str | None
    signature: str = ""
    doc: str = ""
    is_exported: bool = False
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)   # names called inside
    route: str = ""               # HTTP route path if this is an endpoint
    route_method: str = ""


@dataclass
class ParsedImport:
    module: str                   # dotted module or relative
    names: list[str]              # imported symbols
    level: int                    # relative-import depth
    line: int


@dataclass
class PythonFileFacts:
    path: str
    language: str = "python"
    symbols: list[ParsedSymbol] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    syntax_error: str = ""
    module_doc: str = ""


class PythonFileAnalyzer:
    """Parses one Python file into structural facts."""

    def __init__(self, is_test_path: bool = False) -> None:
        self.is_test_path = is_test_path

    def analyze(self, rel_path: str, source: str) -> PythonFileFacts:
        facts = PythonFileFacts(path=rel_path)
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            facts.syntax_error = f"line {exc.lineno}: {exc.msg}"
            # Best-effort: parse ignoring the broken region is not feasible;
            # fall back to line-based symbol scan below.
            facts.symbols.extend(_regex_fallback_symbols(rel_path, source))
            return facts

        facts.module_doc = ast.get_docstring(tree) or ""

        for node in tree.body:
            self._scan_top_level(node, facts, parent=None, class_name=None)
        return facts

    # ------------------------------------------------------------------ #

    def _scan_top_level(self, node: ast.stmt, facts: PythonFileFacts, parent: str | None, class_name: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym = self._function_symbol(node, parent=class_name)
            facts.symbols.append(sym)
            if class_name is None:
                self._collect_calls(node, sym)
        elif isinstance(node, ast.ClassDef):
            sym = ParsedSymbol(
                name=node.name,
                kind="class",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                parent=parent,
                doc=ast.get_docstring(node) or "",
                is_exported=not node.name.startswith("_"),
                decorators=_decorator_names(node.decorator_list),
            )
            facts.symbols.append(sym)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method = self._function_symbol(child, parent=node.name)
                    method.kind = "method"
                    facts.symbols.append(method)
                    self._collect_calls(child, method)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            facts.imports.extend(_import_facts(node))

    def _function_symbol(self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent: str | None) -> ParsedSymbol:
        decorators = _decorator_names(node.decorator_list)
        name = node.name
        is_test = name.startswith("test_") or (self.is_test_path and not name.startswith("_"))
        kind = "test" if is_test else "function"
        route, route_method = _route_from_decorators(node.decorator_list)

        sym = ParsedSymbol(
            name=name,
            kind="route" if route else kind,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parent=parent,
            signature=_signature(node),
            doc=ast.get_docstring(node) or "",
            is_exported=not name.startswith("_"),
            decorators=decorators,
            route=route,
            route_method=route_method,
        )
        return sym

    def _collect_calls(self, node: ast.FunctionDef | ast.AsyncFunctionDef, sym: ParsedSymbol) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                name = _call_name(sub.func)
                if name:
                    sym.calls.append(name)


def _decorator_names(decorator_list: list[ast.expr]) -> list[str]:
    names = []
    for dec in decorator_list:
        if isinstance(dec, ast.Call):
            names.append(_call_name(dec.func) or "")
        elif isinstance(dec, (ast.Name, ast.Attribute)):
            names.append(_attr_name(dec))
        elif isinstance(dec, ast.Constant):
            names.append(str(dec.value))
    return [n for n in names if n]


def _route_from_decorators(decorator_list: list[ast.expr]) -> tuple[str, str]:
    """Detect FastAPI/Flask style route decorators: @app.get('/x')."""
    for dec in decorator_list:
        if isinstance(dec, ast.Call):
            func = dec.func
            method = getattr(func, "attr", "") or ""
            owner = getattr(func, "value", None)
            owner_name = getattr(owner, "id", "") if owner else ""
            if method.lower() in {"get", "post", "put", "patch", "delete", "head", "options"} and dec.args:
                arg = dec.args[0]
                path = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else ""
                return path, method.upper()
            # Flask: @app.route('/x', methods=['POST'])
            if method == "route" and dec.args:
                arg = dec.args[0]
                path = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else ""
                verb = "GET"
                for kw in dec.keywords:
                    if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)) and kw.value.elts:
                        first = kw.value.elts[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            verb = first.value.upper()
                return path, verb
            if method == "route" and owner_name in ("app", "api", "bp", "blueprint") and not dec.args:
                return "", "GET"
    return "", ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({args}){returns}"
    except Exception:
        return node.name


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return _attr_name(func)
    return ""


def _attr_name(node: ast.expr) -> str:
    parts: list[str] = []
    cur: ast.expr | None = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _import_facts(node: ast.Import | ast.ImportFrom) -> list[ParsedImport]:
    if isinstance(node, ast.Import):
        return [
            ParsedImport(module=alias.name, names=[], level=0, line=node.lineno)
            for alias in node.names
        ]
    module = node.module or ""
    return [
        ParsedImport(
            module=module,
            names=[alias.name for alias in node.names],
            level=node.level,
            line=node.lineno,
        )
    ]


def _regex_fallback_symbols(rel_path: str, source: str) -> list[ParsedSymbol]:
    """Line-scan fallback for files with syntax errors (keeps index useful)."""
    import re

    symbols: list[ParsedSymbol] = []
    pattern = re.compile(r"^(\s*)(async\s+def|def|class)\s+([A-Za-z_]\w*)")
    for i, line in enumerate(source.splitlines(), start=1):
        m = pattern.match(line)
        if m:
            indent, kw, name = m.groups()
            kind = "class" if kw == "class" else ("test" if name.startswith("test_") else "function")
            symbols.append(
                ParsedSymbol(
                    name=name, kind=kind, line_start=i, line_end=i,
                    parent=None, signature=f"{kw} {name}",
                    is_exported=not name.startswith("_"),
                )
            )
    return symbols
