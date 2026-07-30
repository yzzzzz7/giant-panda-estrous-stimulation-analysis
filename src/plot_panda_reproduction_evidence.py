from __future__ import annotations

from pathlib import Path
import math
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
MASTER_TSV = (
    BASE
    / "final_master_candidate_evidence"
    / "final_master_reproduction_candidate_evidence.tsv"
)
SELECTION_TSV = BASE / "选择分析" / "selection_results.tsv"
OUTPUT_DIR = BASE / "final_figures"
MANUSCRIPT_DIR = PROJECT / "Final" / "论文图片"

WIDTH_CM = 17.5
HEIGHT_CM = 14.6
CM_PER_INCH = 2.54

TEAL = "#3F6C76"
TEAL_LIGHT = "#C9D7D9"
GOLD = "#F2C94C"
GOLD_LIGHT = "#FAE9A8"
GREY = "#B8B8B8"
LIGHT_GREY = "#E8E8E8"
ROW_GREY = "#F7F7F7"
BLACK = "#000000"
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.2,
        "axes.labelsize": 9.4,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
        "axes.edgecolor": BLACK,
        "axes.linewidth": 0.8,
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
    }
)


def representative_gene(row: pd.Series) -> str:
    prior = str(row.get("literature_prior_symbols", "")).strip()
    if prior:
        return prior.split(";")[0]
    symbols = []
    for item in str(row.get("gene_symbols_by_species", "")).split(";"):
        if ":" in item:
            symbols.append(item.split(":", 1)[1])
    non_locus = [symbol for symbol in symbols if symbol and not symbol.upper().startswith("LOC")]
    if non_locus:
        return non_locus[0]
    if symbols:
        return symbols[0]
    return str(row["Orthogroup"])


def numeric(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def draw_marker(ax, x: float, y: float, kind: str) -> None:
    if kind == "support":
        ax.scatter(x, y, s=30, marker="o", facecolor=TEAL, edgecolor=BLACK, linewidth=0.55, zorder=4)
    elif kind == "nominal":
        ax.scatter(x, y, s=34, marker="D", facecolor=GOLD, edgecolor=BLACK, linewidth=0.55, zorder=4)
    elif kind == "neutral":
        ax.scatter(x, y, s=28, marker="o", facecolor=WHITE, edgecolor=GREY, linewidth=0.8, zorder=4)
    elif kind == "intensified":
        ax.scatter(x, y, s=40, marker="^", facecolor=TEAL, edgecolor=BLACK, linewidth=0.55, zorder=4)
    elif kind == "relaxed":
        ax.scatter(x, y, s=40, marker="v", facecolor=GOLD, edgecolor=BLACK, linewidth=0.55, zorder=4)
    elif kind == "partial":
        ax.scatter(x, y, s=34, marker="D", facecolor=GOLD_LIGHT, edgecolor=BLACK, linewidth=0.65, zorder=4)
    elif kind == "unresolved":
        ax.scatter(x, y, s=34, marker="x", color=BLACK, linewidth=1.05, zorder=4)


def cafe_marker(row: pd.Series) -> str | None:
    if str(row.get("cafe_robust_supported_datasets", "")).strip():
        return "nominal"
    if str(row.get("cafe_selected_dataset", "")).strip():
        return "neutral"
    return None


def absrel_marker(row: pd.Series) -> str | None:
    if str(row.get("sequence_analysis_available", "")) != "1":
        return None
    fdr = numeric(row.get("absrel_fdr"))
    return "support" if math.isfinite(fdr) and fdr < 0.05 else "neutral"


def relax_marker(row: pd.Series) -> str | None:
    if str(row.get("sequence_analysis_available", "")) != "1":
        return None
    fdr = numeric(row.get("relax_fdr"))
    if not math.isfinite(fdr) or fdr >= 0.05:
        return "neutral"
    return "intensified" if str(row.get("relax_direction")) == "intensification" else "relaxed"


def assembly_marker(status: str) -> str:
    if status in {"ALL_EXPECTED_COPIES_CONFIRMED", "CONTRACTION_COPY_COUNT_CONFIRMED"}:
        return "support"
    if status in {
        "DISRUPTED_PSEUDOGENE_CANDIDATE",
        "SOME_COPIES_CONFIRMED",
        "CONTRACTION_NOT_CONFIRMED",
    }:
        return "partial"
    return "unresolved"


def neighborhood_marker(status: str) -> str:
    if status == "CONSERVED_PANDA_LOCUS":
        return "support"
    if status == "SYNTENIC_GAP_SUPPORT":
        return "nominal"
    return "unresolved"


def full_box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.2, color=BLACK)
    ax.grid(False)


master = pd.read_csv(MASTER_TSV, sep="\t", dtype=str).fillna("")
selection = pd.read_csv(SELECTION_TSV, sep="\t")
master["gene"] = master.apply(representative_gene, axis=1)
gene_lookup = dict(zip(master["Orthogroup"], master["gene"]))

grade_order = {"A": 0, "B": 1, "C": 2}
integrated = master[master["master_evidence_grade"].isin(["A", "B", "C"])].copy()
integrated["grade_order"] = integrated["master_evidence_grade"].map(grade_order)
integrated = integrated.sort_values(["grade_order", "Orthogroup"]).reset_index(drop=True)

absrel = selection[selection["absrel_significant_fdr05"].eq(1)].copy()
relax = selection[selection["relax_significant_fdr05"].eq(1)].copy()

assert len(integrated) == 18
assert (integrated["master_evidence_grade"] == "A").sum() == 7
assert (integrated["master_evidence_grade"] == "B").sum() == 6
assert (integrated["master_evidence_grade"] == "C").sum() == 5
assert len(absrel) == 9
assert len(relax) == 7

fig = plt.figure(figsize=(WIDTH_CM / CM_PER_INCH, HEIGHT_CM / CM_PER_INCH))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.48, 1.0],
    left=0.135,
    right=0.985,
    bottom=0.095,
    top=0.965,
    hspace=0.38,
    wspace=0.72,
)

# Panel A. Evidence matrix
ax_a = fig.add_subplot(gs[0, :])
n_rows = len(integrated)
matrix_columns = [
    "CAFE",
    "aBSREL",
    "RELAX",
    "Qinling\nassembly",
    "Sichuan\nassembly",
    "Gene\nneighborhood",
    "Evidence\ngrade",
]
x_positions = np.arange(len(matrix_columns))
y_positions = np.arange(n_rows)[::-1]

ax_a.set_xlim(-1.72, len(matrix_columns) - 0.35)
ax_a.set_ylim(-0.8, n_rows + 1.75)
ax_a.axis("off")

for row_number, y in enumerate(y_positions):
    if row_number % 2 == 1:
        ax_a.add_patch(
            Rectangle(
                (-0.52, y - 0.49),
                len(matrix_columns),
                0.98,
                facecolor=ROW_GREY,
                edgecolor="none",
                zorder=0,
            )
        )

ax_a.add_patch(
    Rectangle(
        (-0.52, -0.5),
        len(matrix_columns),
        n_rows,
        facecolor="none",
        edgecolor=BLACK,
        linewidth=0.8,
        zorder=5,
    )
)

for x, label in zip(x_positions, matrix_columns):
    ax_a.text(x, n_rows + 0.35, label, ha="center", va="bottom", fontsize=8.2, color=BLACK)

priority_genes = {"SPIRE2", "KLK14", "TDRD12", "C18H11orf80", "TEX11"}
grade_fills = {"A": TEAL_LIGHT, "B": GOLD_LIGHT, "C": LIGHT_GREY}

for row_number, (_, row) in enumerate(integrated.iterrows()):
    y = y_positions[row_number]
    gene = row["gene"]
    ax_a.text(
        -0.68,
        y,
        gene,
        ha="right",
        va="center",
        fontsize=8.3,
        fontstyle="italic",
        fontweight="bold" if gene in priority_genes else "normal",
        color=BLACK,
    )

    marker_specs = [
        cafe_marker(row),
        absrel_marker(row),
        relax_marker(row),
        assembly_marker(row["Qinling_validation_status"]),
        assembly_marker(row["Sichuan_validation_status"]),
        neighborhood_marker(row["gene_neighborhood_evidence"]),
    ]
    for x, kind in enumerate(marker_specs):
        if kind:
            draw_marker(ax_a, x, y, kind)

    grade = row["master_evidence_grade"]
    ax_a.add_patch(
        Rectangle(
            (6 - 0.25, y - 0.29),
            0.5,
            0.58,
            facecolor=grade_fills[grade],
            edgecolor=BLACK,
            linewidth=0.55,
            zorder=3,
        )
    )
    ax_a.text(6, y, grade, ha="center", va="center", fontsize=8.2, fontweight="bold", color=BLACK, zorder=4)

grade_breaks = [
    n_rows - (integrated["master_evidence_grade"] == "A").sum() - 0.5,
    (integrated["master_evidence_grade"] == "C").sum() - 0.5,
]
for y in grade_breaks:
    ax_a.plot([-0.52, len(matrix_columns) - 0.52], [y, y], color=BLACK, linewidth=0.7, zorder=5)

legend_y = n_rows + 1.18
legend_items = [
    ("support", "significant / confirmed"),
    ("intensified", "selection intensified"),
    ("relaxed", "selection relaxed"),
    ("nominal", "nominal / syntenic support"),
    ("neutral", "evaluated, no signal"),
    ("unresolved", "unresolved / not confirmed"),
]
legend_x = [-0.30, 1.05, 2.28, 3.43, 4.76, 6.03]
for x, (kind, label) in zip(legend_x, legend_items):
    draw_marker(ax_a, x, legend_y, kind)
    ax_a.text(x + 0.13, legend_y, label, ha="left", va="center", fontsize=7.45, color=BLACK)

ax_a.text(
    -0.072,
    1.03,
    "A",
    transform=ax_a.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    fontweight="bold",
    color=BLACK,
)

# Panel B. aBSREL signals
ax_b = fig.add_subplot(gs[1, 0])
absrel["gene"] = absrel["orthogroup"].map(gene_lookup).fillna(absrel["orthogroup"])
absrel["plot_score"] = absrel["absrel_fdr"].apply(
    lambda value: 16.0 if value <= 0 else min(-math.log10(value), 16.0)
)
absrel = absrel.sort_values(["plot_score", "gene"], ascending=[False, True]).reset_index(drop=True)
y_b = np.arange(len(absrel))

for y, row in zip(y_b, absrel.itertuples()):
    ax_b.hlines(y, 0, row.plot_score, color=TEAL_LIGHT, linewidth=2.0, zorder=1)
    marker = ">" if row.absrel_fdr <= 0 else "o"
    ax_b.scatter(
        row.plot_score,
        y,
        s=38,
        marker=marker,
        facecolor=TEAL,
        edgecolor=BLACK,
        linewidth=0.6,
        zorder=3,
    )

ax_b.axvline(-math.log10(0.05), color=BLACK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
ax_b.text(
    -math.log10(0.05) + 0.18,
    -0.82,
    "FDR = 0.05",
    ha="left",
    va="center",
    fontsize=7.7,
    color=BLACK,
)
ax_b.set_yticks(y_b)
ax_b.set_yticklabels(absrel["gene"])
for label in ax_b.get_yticklabels():
    label.set_fontstyle("italic")
ax_b.invert_yaxis()
ax_b.set_xlim(0, 17.0)
ax_b.set_xlabel(r"$-\log_{10}(\mathrm{FDR})$")
ax_b.set_ylabel("")
full_box(ax_b)
ax_b.text(
    -0.19,
    1.08,
    "B",
    transform=ax_b.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    fontweight="bold",
    color=BLACK,
)

# Panel C. RELAX K
ax_c = fig.add_subplot(gs[1, 1])
relax["gene"] = relax["orthogroup"].map(gene_lookup).fillna(relax["orthogroup"])
relax["plot_log2_k"] = relax["relax_k"].apply(
    lambda value: -6.0 if value <= 0 else max(min(math.log2(value), 6.0), -6.0)
)
relax = relax.sort_values(["plot_log2_k", "gene"], ascending=[False, True]).reset_index(drop=True)
y_c = np.arange(len(relax))

for y, row in zip(y_c, relax.itertuples()):
    color = TEAL if row.relax_direction == "intensification" else GOLD
    light_color = TEAL_LIGHT if row.relax_direction == "intensification" else GOLD_LIGHT
    ax_c.hlines(y, 0, row.plot_log2_k, color=light_color, linewidth=2.0, zorder=1)
    marker = "^" if row.relax_direction == "intensification" else "v"
    if row.relax_k <= 0:
        marker = "<"
    ax_c.scatter(
        row.plot_log2_k,
        y,
        s=42,
        marker=marker,
        facecolor=color,
        edgecolor=BLACK,
        linewidth=0.6,
        zorder=3,
    )

ax_c.axvline(0, color=BLACK, linewidth=0.8, zorder=0)
ax_c.set_yticks(y_c)
ax_c.set_yticklabels(relax["gene"])
for label in ax_c.get_yticklabels():
    label.set_fontstyle("italic")
ax_c.invert_yaxis()
ax_c.set_xlim(-6.6, 6.25)
ax_c.set_xlabel(r"$\log_{2}(K)$")
ax_c.set_ylabel("")
ax_c.text(-4.15, -0.83, "Relaxed selection", ha="center", va="center", fontsize=7.8, color=BLACK)
ax_c.text(3.25, -0.83, "Intensified selection", ha="center", va="center", fontsize=7.8, color=BLACK)
full_box(ax_c)
ax_c.text(
    -0.19,
    1.08,
    "C",
    transform=ax_c.transAxes,
    ha="left",
    va="top",
    fontsize=15,
    fontweight="bold",
    color=BLACK,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

pdf_path = OUTPUT_DIR / "panda_reproduction_evidence_science_style.pdf"
jpg_path = OUTPUT_DIR / "panda_reproduction_evidence_science_style.jpg"
fig.savefig(pdf_path, format="pdf")
fig.savefig(jpg_path, format="jpg", dpi=600)
plt.close(fig)

manuscript_pdf = MANUSCRIPT_DIR / "大熊猫繁殖相关基因比较基因组证据图_Science版.pdf"
manuscript_jpg = MANUSCRIPT_DIR / "大熊猫繁殖相关基因比较基因组证据图_Science版.jpg"
shutil.copy2(pdf_path, manuscript_pdf)
shutil.copy2(jpg_path, manuscript_jpg)

print(f"PDF\t{pdf_path}")
print(f"JPG\t{jpg_path}")
print(f"MANUSCRIPT_PDF\t{manuscript_pdf}")
print(f"MANUSCRIPT_JPG\t{manuscript_jpg}")
