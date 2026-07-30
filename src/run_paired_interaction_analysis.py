from itertools import combinations
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "输入文件"
OUTPUT = ROOT / "输出文件"
FIGURE = ROOT / "图"
TEXT = "#000000"
AI_COLOR = "#F0E3D2"
NM_COLOR = "#B4D6A1"
SUSPECT_BACKGROUND = {"Com_172", "Com_96"}

# Samples from the same male under non-estrus female stimulation (US) and
# estrus female stimulation (ST), transcribed from the provided sample sheet.
PAIRS = [
    ("AI_01", "AI", "all_US_AI_1", "all_ST_AI_1"),
    ("AI_02", "AI", "all_US_AI_2", "all_ST_AI_2"),
    ("AI_03", "AI", "all_US_AI_3", "all_ST_AI_3"),
    ("AI_04", "AI", "all_US_AI_4", "all_ST_AI_4"),
    ("AI_05", "AI", "all_US_AI_5", "all_ST_AI_5"),
    ("AI_06", "AI", "all_US_AI_6", "all_ST_AI_6"),
    ("AI_07", "AI", "all_US_AI_7", "all_ST_AI_7"),
    ("NM_01", "NM", "all_US_NM_1", "all_ST_NM_1"),
    ("NM_02", "NM", "all_US_NM_3", "all_ST_NM_3"),
    ("NM_03", "NM", "all_US_NM_4", "all_ST_NM_4"),
    ("NM_04", "NM", "all_US_NM_5", "all_ST_NM_5"),
    ("NM_05", "NM", "all_US_NM_6", "all_ST_NM_2"),
]


def read_tsv(path):
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig", engine="python")


def exact_group_permutation(ai_values, nm_values):
    """Two-sided exact test for the difference in paired response magnitudes."""
    values = np.r_[ai_values, nm_values]
    n_ai = len(ai_values)
    observed = float(np.mean(nm_values) - np.mean(ai_values))
    null = []
    for indices in combinations(range(len(values)), n_ai):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(indices)] = True
        null.append(float(np.mean(values[~mask]) - np.mean(values[mask])))
    null = np.asarray(null)
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return observed, p_value, len(null)


def bh(values):
    values = pd.to_numeric(values, errors="coerce")
    adjusted = np.full(len(values), np.nan)
    valid = values.notna().to_numpy()
    if valid.any():
        adjusted[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return adjusted


def short_name(name):
    aliases = {
        "Butanoic acid, 2-methyl-": "2-Methylbutyric acid",
        "Pentanoic acid, 4-oxo-, ethyl ester": "Ethyl levulinate",
        "Butanedioic acid, diethyl ester": "Diethyl succinate",
        "9,12,15-Octadecatrienoic acid, methyl ester, (Z,Z,Z)-": "Methyl linolenate (Z,Z,Z)",
        "9,12,15-Octadecatrienoic acid, methyl ester": "Methyl linolenate",
        "Dodecanoic acid, methyl ester": "Methyl dodecanoate",
        "Hexadecanoic acid, methyl ester": "Methyl hexadecanoate",
        "trans-13-Octadecenoic acid, methyl ester": "Methyl octadecenoate",
        "Methyl tetradecanoate": "Methyl tetradecanoate",
    }
    return aliases.get(name, str(name))


def plot_top_candidates(result):
    candidates = result.loc[~result["suspect_background"]].copy()
    candidates = candidates.nsmallest(12, "interaction_P").iloc[::-1]
    labels = [textwrap.fill(short_name(name), width=34, break_long_words=False) for name in candidates["Name"]]

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
    })
    fig, ax = plt.subplots(figsize=(17.5 / 2.54, 10.5 / 2.54))
    ypos = np.arange(len(candidates))
    effects = candidates["interaction_log2_delta_NM_minus_AI"].to_numpy()
    colors = [NM_COLOR if effect > 0 else AI_COLOR for effect in effects]
    ax.hlines(ypos, 0, effects, color="#8A8A8A", linewidth=0.65, zorder=1)
    ax.scatter(effects, ypos, s=38, facecolor=colors, edgecolor=TEXT, linewidth=0.55, zorder=3)
    ax.axvline(0, color="#6F6F6F", linewidth=0.65, linestyle=(0, (3, 3)), zorder=0)
    ax.set_yticks(ypos, labels)
    ax.set_xlabel("Difference in stimulus response, Δ(ST-US): NM - AI (log2 peak area)", fontsize=9)
    ax.set_xlim(float(min(effects.min() - 0.5, -0.8)), float(max(effects.max() + 0.5, 0.8)))
    ax.tick_params(axis="both", labelsize=8, width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(TEXT)
        spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.39, right=0.98, bottom=0.16, top=0.95)
    fig.savefig(FIGURE / "paired_interaction_top_candidates_17.5cm.jpg", dpi=600)
    fig.savefig(FIGURE / "paired_interaction_top_candidates_17.5cm.pdf")
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    FIGURE.mkdir(exist_ok=True)
    intensity = read_tsv(INPUT / "metabolite_peak_area_matrix.tsv")
    annotation = read_tsv(INPUT / "metabolite_annotation.tsv")
    sample_columns = [column for pair in PAIRS for column in pair[2:]]
    missing = [column for column in sample_columns if column not in intensity.columns]
    if missing:
        raise ValueError(f"Missing sample columns: {missing}")

    rows = []
    pair_rows = []
    for _, metabolite in intensity.iterrows():
        values = pd.to_numeric(metabolite[sample_columns], errors="coerce")
        positive = values[np.isfinite(values) & (values > 0)]
        pseudocount = float(positive.min() / 2) if len(positive) else 1e-9
        log_values = np.log2(np.maximum(values.astype(float), 0) + pseudocount)

        deltas = []
        for individual, mating, us_sample, st_sample in PAIRS:
            delta = float(log_values[st_sample] - log_values[us_sample])
            deltas.append((individual, mating, delta))
            pair_rows.append({
                "Compound_ID": metabolite["Compound_ID"], "Individual": individual,
                "Mating_group": mating, "delta_ST_minus_US_log2": delta,
            })
        delta_table = pd.DataFrame(deltas, columns=["Individual", "Mating_group", "delta"])
        ai_delta = delta_table.loc[delta_table.Mating_group.eq("AI"), "delta"].to_numpy()
        nm_delta = delta_table.loc[delta_table.Mating_group.eq("NM"), "delta"].to_numpy()
        interaction, p_value, n_permutations = exact_group_permutation(ai_delta, nm_delta)
        within_ai_p = float(ttest_1samp(ai_delta, popmean=0, nan_policy="omit").pvalue)
        within_nm_p = float(ttest_1samp(nm_delta, popmean=0, nan_policy="omit").pvalue)
        rows.append({
            "Compound_ID": metabolite["Compound_ID"],
            "Name": metabolite.get("Name", ""),
            "Formula": metabolite.get("Formula", ""),
            "RT_min": metabolite.get("RT[min]", np.nan),
            "AI_pairs": len(ai_delta), "NM_pairs": len(nm_delta),
            "AI_mean_delta_ST_minus_US": float(np.mean(ai_delta)),
            "NM_mean_delta_ST_minus_US": float(np.mean(nm_delta)),
            "interaction_log2_delta_NM_minus_AI": interaction,
            "interaction_P": p_value, "exact_permutations": n_permutations,
            "within_AI_paired_t_P": within_ai_p, "within_NM_paired_t_P": within_nm_p,
            "suspect_background": metabolite["Compound_ID"] in SUSPECT_BACKGROUND,
        })

    result = pd.DataFrame(rows)
    annotation_columns = [
        "Compound_ID", "Kegg_ID", "KEGG_pathway", "HMDB_ID", "Lipidmaps_ID",
        "SuperClass(HMDB)", "Class(HMDB)", "SubClass(HMDB)", "CATEGORY(Lipidmaps)",
        "MAIN_CLASS(Lipidmaps)", "SUB_CLASS(Lipidmaps)",
    ]
    result = result.merge(annotation[annotation_columns], on="Compound_ID", how="left")
    result["interaction_FDR"] = bh(result["interaction_P"])
    result["interaction_direction"] = np.where(
        result["interaction_log2_delta_NM_minus_AI"] > 0,
        "stronger ST-associated response in NM", "stronger ST-associated response in AI",
    )
    result = result.sort_values(["interaction_FDR", "interaction_P"], na_position="last").reset_index(drop=True)
    pair_values = pd.DataFrame(pair_rows).merge(result[["Compound_ID", "Name"]], on="Compound_ID", how="left")

    strict = result.loc[(result["interaction_FDR"] < 0.05) & (~result["suspect_background"])].copy()
    nominal = result.loc[(result["interaction_P"] < 0.05) & (~result["suspect_background"])].copy()
    functional_ids = {"Com_7", "Com_65", "Com_60", "Com_71", "Com_66", "Com_3", "Com_86", "Com_12", "Com_8", "Com_26", "Com_38"}
    functional = result.loc[result["Compound_ID"].isin(functional_ids)].copy()
    summary = pd.DataFrame({
        "analysis": [
            "metabolites tested", "complete paired males", "AI complete pairs", "NM complete pairs",
            "interaction candidates, nominal P < 0.05", "interaction candidates, BH-FDR < 0.05",
            "suspect background metabolites excluded from interpretation",
        ],
        "value": [len(result), 12, 7, 5, len(nominal), len(strict), 2],
        "note": [
            "All annotated metabolite features", "Only males sampled under both stimulation conditions",
            "Natural-mating-impaired group", "Natural-mating-capable group",
            "Exact permutation test of the difference in within-male ST-US change", 
            "Primary confirmatory threshold", "Com_172 and Com_96",
        ],
    })

    result.to_csv(OUTPUT / "paired_stimulation_by_mating_interaction_all.csv", index=False, encoding="utf-8-sig")
    nominal.to_csv(OUTPUT / "paired_stimulation_by_mating_nominal_candidates.csv", index=False, encoding="utf-8-sig")
    strict.to_csv(OUTPUT / "paired_stimulation_by_mating_FDR_candidates.csv", index=False, encoding="utf-8-sig")
    functional.to_csv(OUTPUT / "functional_candidates_paired_response_comparison.csv", index=False, encoding="utf-8-sig")
    pair_values.to_csv(OUTPUT / "within_male_ST_minus_US_deltas.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "paired_interaction_summary.csv", index=False, encoding="utf-8-sig")
    plot_top_candidates(result)
    print(summary.to_string(index=False))
    print(result.loc[~result["suspect_background"], ["Compound_ID", "Name", "AI_mean_delta_ST_minus_US", "NM_mean_delta_ST_minus_US", "interaction_log2_delta_NM_minus_AI", "interaction_P", "interaction_FDR"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
