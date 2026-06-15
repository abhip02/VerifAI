"""Render DFA state diagrams for the 4 specs used in the paper appendix.

Specs (from scenic_scenarios/specs.py):
  spec_fast_twice       — co-safety: fast → slow → fast V-shape (absorbing-reject)
  spec_max_speed        — Markovian: never exceed threshold
  spec_completes_turn   — Markovian: never enter turning band
  spec_k_intersection   — non-Markovian K-window: complete turn within K ticks

Visual style matches the existing example (Screenshot 2026-06-05 at 4.32.45 AM):
pastel-filled circles, double-ring border for accepting states, single ring
for absorbing-reject, "start" arrow into the initial state.
"""

import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Pastel palette matching the reference screenshot.
GREEN = ("#d8efe1", "#1f7a4d")   # fill, stroke
BLUE = ("#e1e1f5", "#4a3fb0")
RED = ("#f7e1dc", "#b13c2a")
GRAY = ("#eeeeee", "#555555")
YELLOW = ("#fbf3d6", "#9a7b1f")


def node(name, label, fill, stroke, double=True):
    peripheries = 2 if double else 1
    return (
        f'  {name} [label="{label}", shape=circle, style="filled", '
        f'fillcolor="{fill}", color="{stroke}", fontcolor="{stroke}", '
        f'fontname="Helvetica", peripheries={peripheries}, '
        f'width=1.0, fixedsize=true];\n'
    )


def edge(src, dst, label, color="#555555"):
    return (
        f'  {src} -> {dst} [label="{label}", color="{color}", '
        f'fontcolor="{color}", fontname="Helvetica", fontsize=11];\n'
    )


def header(name, rankdir="LR"):
    return (
        f"digraph {name} {{\n"
        f"  rankdir={rankdir};\n"
        f'  bgcolor="white";\n'
        f'  node [fontname="Helvetica"];\n'
        f'  edge [fontname="Helvetica"];\n'
        f"  start [shape=plaintext, label=\"start\", fontcolor=\"#777777\"];\n"
    )


# --------------------------------------------------------------------------
# 1) spec_fast_twice  (alphabet: slow, mid, fast)
# --------------------------------------------------------------------------
def dot_fast_twice():
    s = header("fast_twice")
    s += node("ok", "ok", *GREEN)
    s += node("was_fast", "was_fast", *BLUE)
    s += node("was_slow", "was_slow", *YELLOW)
    s += node("violated", "violated", *RED, double=False)
    s += edge("start", "ok", "")
    s += edge("ok", "ok", "slow, mid")
    s += edge("ok", "was_fast", "fast")
    s += edge("was_fast", "was_fast", "mid, fast")
    s += edge("was_fast", "was_slow", "slow")
    s += edge("was_slow", "was_slow", "slow, mid")
    s += edge("was_slow", "violated", "fast")
    s += edge("violated", "violated", "any (absorbing)")
    s += "}\n"
    return s


# --------------------------------------------------------------------------
# 2) spec_max_speed  (alphabet: low, high)
# --------------------------------------------------------------------------
def dot_max_speed():
    s = header("max_speed")
    s += node("ok", "ok", *GREEN)
    s += node("violated", "violated", *RED, double=False)
    s += edge("start", "ok", "")
    s += edge("ok", "ok", "low")
    s += edge("ok", "violated", "high")
    s += edge("violated", "violated", "any (absorbing)")
    s += "}\n"
    return s


# --------------------------------------------------------------------------
# 3) spec_completes_turn  (alphabet: straight, turn)
# --------------------------------------------------------------------------
def dot_completes_turn():
    s = header("completes_turn")
    s += node("ok", "ok", *GREEN)
    s += node("violated", "violated", *RED, double=False)
    s += edge("start", "ok", "")
    s += edge("ok", "ok", "straight")
    s += edge("ok", "violated", "turn")
    s += edge("violated", "violated", "any (absorbing)")
    s += "}\n"
    return s


# --------------------------------------------------------------------------
# 4) spec_k_intersection  (alphabet: pre_turn, in_turning, turn_done)
#    K=10 in code; we render in_turn_1, in_turn_2, …, in_turn_K with ellipsis.
# --------------------------------------------------------------------------
def dot_k_intersection():
    s = header("k_intersection", rankdir="LR")
    s += node("ok", "ok", *GREEN)
    s += node("in_turn_1", "in_turn_1", *BLUE)
    s += node("in_turn_2", "in_turn_2", *BLUE)
    s += ('  dots [label="...", shape=plaintext, fontsize=18, '
          'fontcolor="#555555"];\n')
    s += node("in_turn_K", "in_turn_K", *BLUE)
    s += node("completed", "completed", *GREEN)
    s += node("violated", "violated", *RED, double=False)
    s += edge("start", "ok", "")

    s += edge("ok", "ok", "pre_turn")
    s += edge("ok", "completed", "turn_done")
    s += edge("ok", "in_turn_1", "in_turning")

    s += edge("in_turn_1", "in_turn_2", "in_turning")
    s += edge("in_turn_1", "completed", "turn_done")
    s += edge("in_turn_1", "violated", "pre_turn")

    s += edge("in_turn_2", "dots", "in_turning")
    s += edge("in_turn_2", "completed", "turn_done")
    s += edge("in_turn_2", "violated", "pre_turn")

    s += edge("dots", "in_turn_K", "in_turning")

    s += edge("in_turn_K", "violated", "in_turning")
    s += edge("in_turn_K", "completed", "turn_done")
    s += edge("in_turn_K", "violated", "pre_turn")

    s += edge("completed", "completed", "any (absorbing)")
    s += edge("violated", "violated", "any (absorbing)")
    s += "}\n"
    return s


def render(name, dot_src):
    dot_path = OUT / f"{name}.dot"
    png_path = OUT / f"{name}.png"
    pdf_path = OUT / f"{name}.pdf"
    dot_path.write_text(dot_src)
    for fmt, path in [("png", png_path), ("pdf", pdf_path)]:
        subprocess.run(
            ["dot", f"-T{fmt}", str(dot_path), "-o", str(path)],
            check=True,
        )
    print(f"  wrote {png_path.name}, {pdf_path.name}")


def main():
    print("Rendering DFA diagrams →", OUT)
    render("dfa_fast_twice", dot_fast_twice())
    render("dfa_max_speed", dot_max_speed())
    render("dfa_completes_turn", dot_completes_turn())
    render("dfa_k_intersection", dot_k_intersection())


if __name__ == "__main__":
    main()
