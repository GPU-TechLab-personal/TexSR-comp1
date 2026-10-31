#!/usr/bin/env python3
"""L0: is the moved code the old code?

Reads every variants/tests/moves/*.json (one file per area, so parallel work
never edits the same table) and checks three kinds of entry. Pure `ast`, no
imports of the code under test, so it runs on the host (any Python 3.8+).

  "moves"   a whole file copied. Every top-level def/class of the old file must
            be in the new one with an identical AST (docstrings excluded), and the
            remaining module-level statements must match too - except imports,
            sys.path edits, and the assignments named in "module_allow" (the path
            constants that had to change).

  "functions"  one function (or class) moved on its own:
            "old.py:name" -> "new.py:name" or "new.py:Class.method". Identical
            AST, docstring and the def's own name excluded.

  "decomposed"  a function that was split up (deeptex_train.main, the exp2 train
            loop ...). AST identity cannot hold, so the check is coverage: every
            code line of the old function (comments and blank lines dropped,
            whitespace normalised) must occur as a line somewhere in the listed
            new files, or be listed in "rewritten" (old line -> the new line that
            replaced it, which must itself exist) or "dropped" (with the reason).
            A line that is in none of the three fails.

Exit 0 = every entry PASS. --report writes the per-entry table as markdown.
"""

import argparse
import ast
import glob
import io
import json
import os
import sys
import tokenize

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     os.pardir, os.pardir))


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


class _Imports(ast.NodeTransformer):
    """`import grouped_dataset` and `from variants.deeptex.render_loss import
    grouped_dataset` bind the same name; only the bound names are compared."""

    def _canon(self, node):
        names = sorted(a.asname or a.name.split(".")[0] for a in node.names)
        return ast.copy_location(ast.Expr(ast.Constant("import:" + ",".join(names))), node)

    visit_Import = _canon
    visit_ImportFrom = _canon


def parse(rel):
    return _Imports().visit(strip_docstrings(ast.parse(read(rel), filename=rel)))


def dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def defs(tree):
    return {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def is_import_or_path(stmt):
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return True
    if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and str(stmt.value.value).startswith("import:")):
        return True
    # sys.path.insert(...) / sys.path.append(...), bare or under an `if`.
    src = dump(stmt)
    return "attr='path'" in src and "id='sys'" in src and (
        "attr='insert'" in src or "attr='append'" in src)


def assigned_names(stmt):
    out = set()
    targets = []
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        targets = [stmt.target]
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.add(n.id)
    return out


def module_rest(tree, allow):
    rest = []
    for s in tree.body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if is_import_or_path(s):
            continue
        if assigned_names(s) & set(allow):
            continue
        rest.append(dump(s))
    return rest


def find(tree, dotted):
    node = None
    scope = tree.body
    for part in dotted.split("."):
        node = next((n for n in scope if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == part), None)
        if node is None:
            return None
        scope = node.body
    return node


def body_dump(node):
    """The def without its name, so a moved function may be renamed."""
    saved, node.name = node.name, "_"
    try:
        return dump(node)
    finally:
        node.name = saved


# ------------------------------------------------------------------ line coverage

def code_lines(src):
    """Source lines with comments removed and whitespace collapsed, blank ones
    dropped. tokenize, not split('#'): a '#' inside a string is code."""
    comments = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comments.setdefault(tok.start[0], tok.start[1])
    except (tokenize.TokenError, IndentationError):
        pass
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        if i in comments:
            line = line[:comments[i]]
        norm = " ".join(line.split())
        if norm:
            out.append(norm)
    return out


def function_source(rel, name):
    src = read(rel)
    tree = ast.parse(src)
    node = find(tree, name)
    if node is None:
        return None
    lines = src.splitlines()
    # The docstring is documentation, not behaviour.
    start = node.lineno
    body0 = node.body[0]
    if (isinstance(body0, ast.Expr) and isinstance(getattr(body0, "value", None), ast.Constant)
            and isinstance(body0.value.value, str)):
        doc = set(range(body0.lineno, body0.end_lineno + 1))
    else:
        doc = set()
    keep = [lines[i - 1] for i in range(start, node.end_lineno + 1) if i not in doc]
    return "\n".join(keep)


def norm(s):
    return " ".join(s.split())


# ------------------------------------------------------------------------ checks

def check_move(e):
    old, new = parse(e["old"]), parse(e["new"])
    od, nd = defs(old), defs(new)
    problems = []
    for name, node in od.items():
        if name not in nd:
            problems.append("missing def %s" % name)
        elif dump(node) != dump(nd[name]):
            problems.append("def %s differs" % name)
    allow = e.get("module_allow", [])
    a, b = module_rest(old, allow), module_rest(new, allow)
    if a != b:
        missing = [x for x in a if x not in b]
        extra = [x for x in b if x not in a]
        problems.append("module-level statements differ (%d only in old, %d only in new)"
                        % (len(missing), len(extra)))
    extra_defs = sorted(set(nd) - set(od))
    detail = "%d defs identical" % (len(od) - sum(p.startswith(("missing", "def ")) for p in problems))
    if extra_defs:
        detail += "; new-only: %s" % ", ".join(extra_defs)
    return problems, detail


def check_function(old_ref, new_ref):
    of, on = old_ref.split(":")
    nf, nn = new_ref.split(":")
    a, b = find(parse(of), on), find(parse(nf), nn)
    if a is None:
        return ["old %s not found" % old_ref], ""
    if b is None:
        return ["new %s not found" % new_ref], ""
    if body_dump(a) != body_dump(b):
        return ["%s != %s" % (old_ref, new_ref)], ""
    return [], "identical"


def check_decomposed(old_ref, spec):
    of, on = old_ref.split(":")
    src = function_source(of, on)
    if src is None:
        return ["old %s not found" % old_ref], ""
    pool = set()
    for rel in spec["into"]:
        pool.update(code_lines(read(rel)))
    rewritten = {norm(k): norm(v) for k, v in spec.get("rewritten", {}).items()}
    dropped = {norm(k): why for k, why in spec.get("dropped", {}).items()}
    problems = []
    n_verbatim = n_rew = n_drop = 0
    lines = code_lines(src)
    for ln in lines[1:]:                       # [0] is the def line itself
        if ln in pool:
            n_verbatim += 1
        elif ln in rewritten:
            if rewritten[ln] not in pool:
                problems.append("rewritten target not found: %r -> %r" % (ln, rewritten[ln]))
            n_rew += 1
        elif ln in dropped:
            n_drop += 1
        else:
            problems.append("unaccounted line: %r" % ln)
    unused = [k for k in list(rewritten) + list(dropped) if k not in lines]
    if unused:
        problems += ["stale table entry (not in the old function): %r" % k for k in unused]
    return problems, "%d lines: %d verbatim, %d rewritten, %d dropped" % (
        len(lines) - 1, n_verbatim, n_rew, n_drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", nargs="*", default=sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "moves", "*.json"))))
    ap.add_argument("--report", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    spec = {"moves": [], "functions": {}, "decomposed": {}}
    for path in args.moves:
        with open(path, encoding="utf-8") as f:
            part = json.load(f)
        spec["moves"] += part.get("moves", [])
        for k in ("functions", "decomposed"):
            clash = set(spec[k]) & set(part.get(k, {}))
            if clash:
                sys.exit("L0: %s listed twice (%s): %s" % (k, path, sorted(clash)))
            spec[k].update(part.get(k, {}))
    rows, n_fail = [], 0
    for e in spec.get("moves", []):
        probs, detail = check_move(e)
        rows.append(("move", "%s -> %s" % (e["old"], e["new"]), probs, detail))
    for old_ref, new_ref in spec.get("functions", {}).items():
        probs, detail = check_function(old_ref, new_ref)
        rows.append(("function", "%s -> %s" % (old_ref, new_ref), probs, detail))
    for old_ref, d in spec.get("decomposed", {}).items():
        probs, detail = check_decomposed(old_ref, d)
        rows.append(("decomposed", "%s -> %s" % (old_ref, ", ".join(d["into"])), probs, detail))

    out = ["| kind | entry | verdict | detail |", "|---|---|---|---|"]
    for kind, entry, probs, detail in rows:
        v = "PASS" if not probs else "FAIL"
        n_fail += bool(probs)
        print("%-4s %-10s %s  (%s)" % (v, kind, entry, detail))
        shown = probs if args.verbose else probs[:8]
        for p in shown:
            print("       - %s" % p)
        if len(probs) > len(shown):
            print("       ... %d more (-v)" % (len(probs) - len(shown)))
        out.append("| %s | `%s` | %s | %s%s |" % (
            kind, entry, v, detail, "" if not probs else "<br>" + "<br>".join(
                p.replace("|", "\\|") for p in probs[:20])))
    print("L0 static: %d entries, %d FAIL" % (len(rows), n_fail))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
