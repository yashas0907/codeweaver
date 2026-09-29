"""Structural (non-AST where unavailable) analysis for non-Python languages.

Uses line-oriented parsing tuned per language family — good enough for a
repository map (symbols, imports, tests) while staying dependency-free.
Architecture is extensible: add a LanguageParser to PARSERS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GenericSymbol:
    name: str
    kind: str
    line_start: int
    line_end: int
    signature: str = ""
    parent: str | None = None


@dataclass
class GenericFacts:
    path: str
    language: str
    symbols: list[GenericSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


# --- JS / TS --------------------------------------------------------------

_JS_SYMBOL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function\s*\*?|class)\s+([A-Za-z_$][\w$]*)"
)
_JS_ARROW_OR_CONST = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
)
_JS_METHOD = re.compile(
    r"^\s+(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{"
)
_JS_IMPORT = re.compile(
    r"^\s*import\s+(?:[\w{}\s,*$]+\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE
)
_JS_REQUIRE = re.compile(r"=\s*require\(['\"]([^'\"]+)['\"]\)")


def parse_js_like(rel_path: str, source: str, language: str) -> GenericFacts:
    facts = GenericFacts(path=rel_path, language=language)
    lines = source.splitlines()
    current_class: str | None = None
    class_indent = 0
    for i, line in enumerate(lines, start=1):
        m = _JS_SYMBOL.match(line)
        if m:
            name = m.group(1)
            kind = "class" if "class" in line.split(name)[0] else ("test" if ".test." in rel_path or ".spec." in rel_path else "function")
            if kind == "class":
                current_class, class_indent = name, len(line) - len(line.lstrip())
            facts.symbols.append(GenericSymbol(name=name, kind=kind, line_start=i, line_end=i, signature=line.strip()[:120]))
            continue
        if current_class is None:
            m2 = _JS_ARROW_OR_CONST.match(line)
            if m2:
                facts.symbols.append(GenericSymbol(name=m2.group(1), kind="function", line_start=i, line_end=i, signature=line.strip()[:120]))
        else:
            indent = len(line) - len(line.lstrip())
            m3 = _JS_METHOD.match(line)
            if m3 and indent > class_indent:
                facts.symbols.append(GenericSymbol(name=m3.group(1), kind="method", line_start=i, line_end=i, parent=current_class, signature=line.strip()[:120]))
            elif m := re.match(r"^\s*\}\s*;?\s*$", line) and indent <= class_indent:
                current_class = None
    for m in _JS_IMPORT.finditer(source):
        facts.imports.append(m.group(1))
    for m in _JS_REQUIRE.finditer(source):
        facts.imports.append(m.group(1))
    return facts


# --- Go -------------------------------------------------------------------

_GO_FUNC = re.compile(r"^func\s+(?:\((\w+)\s+\*?[\w./]+\)\s*)?([A-Za-z_]\w*)\s*\(")
_GO_STRUCT = re.compile(r"^type\s+([A-Za-z_]\w*)\s+struct\b")
_GO_IMPORT_BLOCK = re.compile(r"import\s*\(([^)]*)\)", re.DOTALL)
_GO_IMPORT_SINGLE = re.compile(r'^import\s+(?:\w+\s+)?"([^"]+)"')


def parse_go(rel_path: str, source: str) -> GenericFacts:
    facts = GenericFacts(path=rel_path, language="go")
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _GO_FUNC.match(line):
            receiver, name = m.groups()
            kind = "method" if receiver else ("test" if name.startswith("Test") else "function")
            facts.symbols.append(GenericSymbol(name=name, kind=kind, line_start=i, line_end=i, parent=receiver or None, signature=line.strip()[:120]))
        elif m := _GO_STRUCT.match(line):
            facts.symbols.append(GenericSymbol(name=m.group(1), kind="class", line_start=i, line_end=i, signature=line.strip()[:120]))
    if m := _GO_IMPORT_BLOCK.search(source):
        for line in m.group(1).splitlines():
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                facts.imports.append(line.strip('"'))
    for m in _GO_IMPORT_SINGLE.finditer(source):
        facts.imports.append(m.group(1))
    return facts


# --- Java / Kotlin / C# ---------------------------------------------------

_JVM_SYMBOL = re.compile(
    r"^\s*(?:public|private|protected|internal|static|\s)*\s*"
    r"(?:final\s+|abstract\s+|async\s+)*"
    r"(class|interface|enum)\s+([A-Za-z_]\w*)"
)
_JVM_METHOD = re.compile(
    r"^\s*(?:public|private|protected|internal)(?:\s+static)?\s+"
    r"[\w<>\[\],.?]+\s+([A-Za-z_]\w*)\s*\([^)]*\)\s*\{?"
)
_JVM_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;")


def parse_jvm_like(rel_path: str, source: str, language: str) -> GenericFacts:
    facts = GenericFacts(path=rel_path, language=language)
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _JVM_SYMBOL.match(line):
            facts.symbols.append(GenericSymbol(name=m.group(2), kind=m.group(1).lower() if m.group(1) != "class" else "class", line_start=i, line_end=i, signature=line.strip()[:120]))
        elif m := _JVM_METHOD.match(line):
            name = m.group(1)
            if name not in ("if", "for", "while", "switch", "catch", "return", "new"):
                kind = "test" if name.startswith("test") or rel_path.lower().endswith(("test.java", "tests.kt")) else "method"
                facts.symbols.append(GenericSymbol(name=name, kind=kind, line_start=i, line_end=i, signature=line.strip()[:120]))
        elif m := _JVM_IMPORT.match(line):
            facts.imports.append(m.group(1))
    return facts


# --- Shell / Ruby / Rust / C-family ----------------------------------------

_RUBY_DEF = re.compile(r"^\s*def\s+([\w.?!=]+)")
_RUBY_CLASS = re.compile(r"^\s*class\s+([A-Z]\w*)")
_RUBY_REQUIRE = re.compile(r"^\s*require(?:_relative)?\s+['\"]([^'\"]+)['\"]")

_RUST_FN = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)")
_RUST_STRUCT = re.compile(r"^\s*(?:pub\s+)?struct\s+([A-Za-z_]\w*)")
_RUST_USE = re.compile(r"^\s*use\s+([\w:{},\s]+);")

_C_FAMILY_FUNC = re.compile(r"^[\w\*][\w\s\*]*\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{?\s*$")
_C_FAMILY_INCLUDE = re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]')

_SHELL_FUNC = re.compile(r"^(?:function\s+)?([A-Za-z_]\w*)\s*\(\)\s*\{")


def parse_ruby(rel_path: str, source: str) -> GenericFacts:
    facts = GenericFacts(path=rel_path, language="ruby")
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _RUBY_CLASS.match(line):
            facts.symbols.append(GenericSymbol(name=m.group(1), kind="class", line_start=i, line_end=i))
        elif m := _RUBY_DEF.match(line):
            name = m.group(1)
            kind = "test" if name.startswith("test_") else "method"
            facts.symbols.append(GenericSymbol(name=name, kind=kind, line_start=i, line_end=i))
        elif m := _RUBY_REQUIRE.match(line):
            facts.imports.append(m.group(1))
    return facts


def parse_rust(rel_path: str, source: str) -> GenericFacts:
    facts = GenericFacts(path=rel_path, language="rust")
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _RUST_FN.match(line):
            name = m.group(1)
            kind = "test" if "mod tests" in source[:i * 0] or name.startswith("test_") else "function"
            facts.symbols.append(GenericSymbol(name=name, kind=kind, line_start=i, line_end=i))
        elif m := _RUST_STRUCT.match(line):
            facts.symbols.append(GenericSymbol(name=m.group(1), kind="class", line_start=i, line_end=i))
        elif m := _RUST_USE.match(line):
            facts.imports.append(m.group(1).split("{")[0].strip())
    return facts


def parse_c_family(rel_path: str, source: str, language: str) -> GenericFacts:
    facts = GenericFacts(path=rel_path, language=language)
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _C_FAMILY_FUNC.match(line):
            facts.symbols.append(GenericSymbol(name=m.group(1), kind="function", line_start=i, line_end=i))
        elif m := _C_FAMILY_INCLUDE.match(line):
            facts.imports.append(m.group(1))
    return facts


def parse_shell(rel_path: str, source: str) -> GenericFacts:
    facts = GenericFacts(path=rel_path, language="shell")
    for i, line in enumerate(source.splitlines(), start=1):
        if m := _SHELL_FUNC.match(line):
            facts.symbols.append(GenericSymbol(name=m.group(1), kind="function", line_start=i, line_end=i))
    return facts


PARSERS = {
    "javascript": lambda p, s: parse_js_like(p, s, "javascript"),
    "typescript": lambda p, s: parse_js_like(p, s, "typescript"),
    "go": parse_go,
    "java": lambda p, s: parse_jvm_like(p, s, "java"),
    "kotlin": lambda p, s: parse_jvm_like(p, s, "kotlin"),
    "csharp": lambda p, s: parse_jvm_like(p, s, "csharp"),
    "ruby": parse_ruby,
    "rust": parse_rust,
    "c": lambda p, s: parse_c_family(p, s, "c"),
    "cpp": lambda p, s: parse_c_family(p, s, "cpp"),
    "shell": parse_shell,
}


def parse_generic(rel_path: str, source: str, language: str) -> GenericFacts | None:
    parser = PARSERS.get(language)
    if parser is None:
        return None
    try:
        return parser(rel_path, source)
    except Exception:
        return None
