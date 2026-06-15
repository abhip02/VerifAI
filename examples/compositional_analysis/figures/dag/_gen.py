"""Render Scenic-DAG figures for the 5 appendix experiment sections.

Style mirrors the reference screenshot:
  * Rounded rectangle for `Main` and primitive-scenario invocations.
  * Lavender filled rectangles for `do` / `do choose` / `do shuffle`
    operator nodes.
  * Dashed-bordered circles for choose/shuffle branch labels with weights.
  * Solid black arrows for parent → operator and operator → primitive.
  * Dashed blue arrow between sibling operators to indicate sequencing
    inside a `compose:` block.
"""

import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent

LAV_FILL = "#dcd5f0"
LAV_STROKE = "#5a4ea8"
BOX_STROKE = "#5a4ea8"
DASH_STROKE = "#5a4ea8"
SEQ_BLUE = "#5a8fd6"


def header(name):
    return (
        f"digraph {name} {{\n"
        f"  rankdir=TB;\n"
        f'  bgcolor="white";\n'
        f"  nodesep=0.35;\n"
        f"  ranksep=0.45;\n"
        f'  node [fontname="Helvetica"];\n'
        f'  edge [fontname="Helvetica", fontsize=11];\n'
    )


def main_node(name="Main"):
    return (
        f'  {name} [label="{name}", shape=box, style="rounded", '
        f'color="{BOX_STROKE}", fontcolor="black", width=1.0];\n'
    )


def scenario_box(node_id, label):
    return (
        f'  {node_id} [label="{label}", shape=box, style="rounded", '
        f'color="{BOX_STROKE}", fontcolor="black", width=0.7];\n'
    )


def op_node(node_id, label):
    return (
        f'  {node_id} [label="{label}", shape=box, style="rounded,filled", '
        f'fillcolor="{LAV_FILL}", color="{LAV_STROKE}", fontcolor="black"];\n'
    )


def branch_circle(node_id, label, width=0.55):
    return (
        f'  {node_id} [label="{label}", shape=ellipse, '
        f'style="dashed", color="{DASH_STROKE}", fontcolor="black", '
        f'width={width}, height={width}];\n'
    )


def weight_node(node_id, text):
    return (
        f'  {node_id} [label="{text}", shape=plaintext, '
        f'fontcolor="#777777", fontsize=11];\n'
    )


def edge(src, dst, style="solid", color="black", arrow="normal", label=""):
    lab = f', label="{label}"' if label else ""
    return (
        f'  {src} -> {dst} [style={style}, color="{color}", '
        f'arrowhead={arrow}{lab}];\n'
    )


def invisible_edge(src, dst):
    return f'  {src} -> {dst} [style=invis];\n'


# --------------------------------------------------------------------------
# A.1 seq_CSXS  — Main: do C(); do S(); do X(); do S()
# --------------------------------------------------------------------------
def dot_seq_CSXS():
    s = header("seq_CSXS")
    s += main_node()
    for i, prim in enumerate(["C", "S", "X", "S"]):
        op = f"op{i}"
        leaf = f"leaf{i}"
        s += op_node(op, "do")
        s += scenario_box(leaf, prim)
        s += edge("Main", op)
        s += edge(op, leaf)
    # sequencing arrows between siblings
    for i in range(3):
        s += edge(f"op{i}", f"op{i+1}", style="dashed", color=SEQ_BLUE)
    # keep operators on one row
    s += "  { rank=same; op0; op1; op2; op3; }\n"
    s += "  { rank=same; leaf0; leaf1; leaf2; leaf3; }\n"
    s += "}\n"
    return s


# --------------------------------------------------------------------------
# A.2 / A.3 native_shuffle — Main: do S();  do shuffle { C:1, X:1, O:1 }
# --------------------------------------------------------------------------
def dot_native_shuffle():
    s = header("native_shuffle")
    s += main_node()
    s += op_node("opS", "do")
    s += scenario_box("leafS", "S")
    s += op_node("opShuf", "do shuffle")

    s += edge("Main", "opS")
    s += edge("Main", "opShuf")
    s += edge("opS", "leafS")
    s += edge("opS", "opShuf", style="dashed", color=SEQ_BLUE)

    for prim in ["C", "X", "O"]:
        s += branch_circle(f"br{prim}", prim)
        s += weight_node(f"w{prim}", "1")
        s += scenario_box(f"leaf{prim}", prim)
        s += edge("opShuf", f"br{prim}", color="#777777")
        s += edge(f"br{prim}", f"leaf{prim}", color="#777777")
        s += f'  br{prim} -> w{prim} [style=invis];\n'

    s += "  { rank=same; opS; opShuf; }\n"
    s += "  { rank=same; brC; brX; brO; }\n"
    s += "  { rank=same; leafS; leafC; leafX; leafO; }\n"
    s += "}\n"
    return s


# --------------------------------------------------------------------------
# A.4 / A.5 maneuver_choose — Main: do choose { TurnL:1, TurnR:1, Straight:1 }
# --------------------------------------------------------------------------
def dot_maneuver_choose():
    s = header("maneuver_choose")
    s += main_node()
    s += op_node("opChoose", "do choose")
    s += edge("Main", "opChoose")

    for prim, key in [("TurnL", "L"), ("TurnR", "R"), ("Straight", "St")]:
        s += branch_circle(f"br{key}", prim, width=1.0)
        s += weight_node(f"w{key}", "1")
        s += scenario_box(f"leaf{key}", prim)
        s += edge("opChoose", f"br{key}", color="#777777")
        s += edge(f"br{key}", f"leaf{key}", color="#777777")
        s += f'  br{key} -> w{key} [style=invis];\n'

    s += "  { rank=same; brL; brR; brSt; }\n"
    s += "  { rank=same; leafL; leafR; leafSt; }\n"
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
    print("Rendering Scenic-DAG diagrams →", OUT)
    # A.1 — Fast Twice CSXS uses seq_CSXS
    render("dag_A1_fast_twice_CSXS", dot_seq_CSXS())
    # A.2 — Fast Twice Shuffle uses native_shuffle
    render("dag_A2_fast_twice_shuffle", dot_native_shuffle())
    # A.3 — Max Speed Shuffle uses native_shuffle
    render("dag_A3_max_speed_shuffle", dot_native_shuffle())
    # A.4 — 4-Way Intersection Choose uses maneuver_choose
    render("dag_A4_intersection_choose", dot_maneuver_choose())
    # A.5 — 4-Way Intersection Choose (Bounded-Time Turn)
    render("dag_A5_intersection_choose_bounded", dot_maneuver_choose())


if __name__ == "__main__":
    main()
