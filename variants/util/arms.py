"""The arm registry (variants/arms.json): one place that says what an SR arm is.

    from variants.util import arms
    arms.display_name("render_r1_40k")   # 'DeepTextures + render loss (R1) 40k'
    arms.column_key("matsynth_all")      # 'dt_matsynth_all'
    arms.pairs(["matsynth_all", "render_r0a_40k", "render_r1_40k"])

A tag is <base>[_<NN>k]; the registry entry of <base> applies. A tag the
registry does not list is still accepted everywhere: it shows as itself, has no
reference and no pairs, and belongs to DEFAULT_METHOD (the evaluators were
written for DeepTextures arms, and its column key prefix keeps existing report
keys). Adding an arm of a new method = one entry under "methods" and one under
"arms"; nothing else names arms.

stdlib only: dataprep, matup, deeptex and exp images all import it.
"""
import json
import os
import re

REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arms.json")
DEFAULT_METHOD = "deeptex"
STEP_RE = re.compile(r"^(.+)_(\d+k)$")

_cache = {}


def registry(path=REGISTRY):
    if path not in _cache:
        with open(path, encoding="utf-8") as fh:
            _cache[path] = json.load(fh)
    return _cache[path]


def split_tag(tag):
    """'render_r1_40k' -> ('render_r1', '40k'); no suffix -> (tag, '')."""
    m = STEP_RE.match(tag)
    return (m.group(1), m.group(2)) if m else (tag, "")


def step_key(step):
    """'40k' -> 40, '' -> -1 (for sorting)."""
    return int(step[:-1]) if step else -1


def entry(tag):
    """The registry entry of the tag's base, or None."""
    return registry()["arms"].get(split_tag(tag)[0])


def method(tag):
    e = entry(tag)
    return e["method"] if e else DEFAULT_METHOD


def method_info(tag):
    return registry()["methods"][method(tag)]


def method_display(tag):
    """'DeepTextures' - the method name, without the arm."""
    return method_info(tag)["display"]


def column_key(tag):
    """The arm's column key in reports: method key_prefix + tag ('dt_render_r1_40k')."""
    return method_info(tag)["key_prefix"] + tag


def base_display(tag):
    """The row name of the tag's base, without a step; an unknown base shows as itself."""
    base = split_tag(tag)[0]
    e = registry()["arms"].get(base)
    return e["display"] if e else base


def display_name(tag):
    """Row name: the registry display + ' <step>'; an unknown tag shows as itself."""
    base, step = split_tag(tag)
    e = registry()["arms"].get(base)
    if not e:
        return tag
    return e["display"] + (" " + step if step else "")


def label_lines(name):
    """One or two lines: break before the bracket ("... (R1) 40k")."""
    i = name.find(" (")
    if i <= 0 or len(name) <= 18:
        return [name]
    return [name[:i], name[i + 1:]]


def reference(tag):
    """The base tag this arm is compared against, or None."""
    e = entry(tag)
    return e.get("reference") if e else None


def head_to_head_bases():
    """[(a, b)] base tags, in report order."""
    return [tuple(p) for p in registry()["head_to_head"]]


def pairs(tags):
    """[(a, b)] of the given tags (same step on both sides) for every registered
    head-to-head pair, in registry order then step order."""
    by = {split_tag(t): t for t in tags}
    steps = sorted({s for _, s in by}, key=step_key)
    out = []
    for s in steps:
        for a, b in head_to_head_bases():
            if (a, s) in by and (b, s) in by:
                out.append((by[(a, s)], by[(b, s)]))
    return out


def first_pair(tags):
    """The first registered head-to-head pair among `tags` (exact tags, no step
    matching), or None."""
    present = set(tags)
    for a, b in head_to_head_bases():
        if a in present and b in present:
            return a, b
    return None
