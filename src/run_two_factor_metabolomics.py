from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input" / "two_factor"
TABLE = ROOT / "results" / "two_factor"


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="	", engine="python", encoding="utf-8-sig")


def bh(values):
    values = pd.to_numeric(values, errors="coerce")
    out = np.full(len(values), np.nan, dtype=float)
    ok = values.notna().to_numpy()
    if ok.any():
        out[ok] = multipletests(values[ok], method="fdr_bh")[1]
    return out


def main():
    TABLE.mkdir(parents=True, exist_ok=True)

    intensity = read_table(INPUT / "meta_intensity_all.xls")
    metadata = read_table(INPUT / "sam_infor_all.xls")
    annotation = read_table(
        INPUT / "meta_intensity_all_hmdb_kegg_lipidmaps.xls"
    )

    sample_cols = metadata["Sample name"].tolist()
    value_cols = [c for c in sample_cols if c in intensity.columns]
    if len(value_cols) != 26:
        raise ValueError(f"Expected 26 sample columns, found {len(value_cols)}")

    meta = metadata.rename(columns={"Sample name": "Sample", "Group name": "Group"})
    group_map = {
        "US_AI": ("US", "AI"),
        "ST_AI": ("ST", "AI"),
        "US_NM": ("US", "NM"),
        "ST_NM": ("ST", "NM"),
    }
    meta[["Stimulation", "Mating"]] = meta["Group"].map(group_map).apply(pd.Series)
    meta["Stimulation"] = pd.Categorical(meta["Stimulation"], categories=["US", "ST"])
    meta["Mating"] = pd.Categorical(meta["Mating"], categories=["AI", "NM"])

    rows = []
    for _, rec in intensity.iterrows():
        y = pd.to_numeric(rec[value_cols], errors="coerce").to_numpy(dtype=float)
        positive = y[np.isfinite(y) & (y > 0)]
        pseudocount = float(np.nanmin(positive) / 2) if positive.size else 1e-9
        # Peak areas are analyzed on a log2 scale. The small pseudocount only handles zeros.
        y_log2 = np.log2(np.maximum(y, 0) + pseudocount)
        dat = meta.copy()
        dat["Abundance"] = y_log2
        dat = dat.dropna(subset=["Abundance"])

        base = {
            "Compound_ID": rec.get("Compound_ID", ""),
            "Name": rec.get("Name", ""),
            "Formula": rec.get("Formula", ""),
            "RT_min": rec.get("RT[min]", np.nan),
            "n": len(dat),
            "US_AI_mean_log2": dat.loc[dat.Group == "US_AI", "Abundance"].mean(),
            "ST_AI_mean_log2": dat.loc[dat.Group == "ST_AI", "Abundance"].mean(),
            "US_NM_mean_log2": dat.loc[dat.Group == "US_NM", "Abundance"].mean(),
            "ST_NM_mean_log2": dat.loc[dat.Group == "ST_NM", "Abundance"].mean(),
        }
        base["stim_log2FC_AI"] = base["ST_AI_mean_log2"] - base["US_AI_mean_log2"]
        base["stim_log2FC_NM"] = base["ST_NM_mean_log2"] - base["US_NM_mean_log2"]
        base["mating_log2FC_US"] = base["US_NM_mean_log2"] - base["US_AI_mean_log2"]
        base["mating_log2FC_ST"] = base["ST_NM_mean_log2"] - base["ST_AI_mean_log2"]
        base["interaction_log2FC"] = base["stim_log2FC_NM"] - base["stim_log2FC_AI"]

        try:
            model = smf.ols(
                "Abundance ~ C(Stimulation, Treatment(reference='US')) * "
                "C(Mating, Treatment(reference='AI'))",
                data=dat,
            ).fit()
            table = anova_lm(model, typ=3)
            residual_ss = float(table.loc["Residual", "sum_sq"])
            for key, label in [
                ("C(Stimulation, Treatment(reference='US'))", "stimulation"),
                ("C(Mating, Treatment(reference='AI'))", "mating"),
                ("C(Stimulation, Treatment(reference='US')):C(Mating, Treatment(reference='AI'))", "interaction"),
            ]:
                base[f"{label}_F"] = float(table.loc[key, "F"])
                base[f"{label}_P"] = float(table.loc[key, "PR(>F)"])
                ss = float(table.loc[key, "sum_sq"])
                base[f"{label}_partial_eta2"] = ss / (ss + residual_ss) if residual_ss > 0 else np.nan
        except Exception:
            for label in ["stimulation", "mating", "interaction"]:
                base[f"{label}_F"] = np.nan
                base[f"{label}_P"] = np.nan
                base[f"{label}_partial_eta2"] = np.nan
        rows.append(base)

    result = pd.DataFrame(rows)
    ann_cols = [
        "Compound_ID",
        "Kegg_ID",
        "KEGG_pathway",
        "HMDB_ID",
        "Lipidmaps_ID",
        "SuperClass(HMDB)",
        "Class(HMDB)",
        "SubClass(HMDB)",
        "CATEGORY(Lipidmaps)",
        "MAIN_CLASS(Lipidmaps)",
        "SUB_CLASS(Lipidmaps)",
    ]
    result = result.merge(annotation[ann_cols], on="Compound_ID", how="left")
    for label in ["stimulation", "mating", "interaction"]:
        result[f"{label}_FDR"] = bh(result[f"{label}_P"])

    result["interaction_direction"] = np.where(
        result["interaction_log2FC"] > 0,
        "stronger stimulation response in NM",
        "stronger stimulation response in AI",
    )
    result = result.sort_values(["interaction_FDR", "interaction_P"], na_position="last")

    strict = result[result["interaction_FDR"] < 0.05].copy()
    exploratory = result[
        (result["interaction_P"] < 0.05)
        & (result["interaction_log2FC"].abs() >= np.log2(1.2))
    ].copy()

    summary = pd.DataFrame(
        {
            "analysis": [
                "metabolites analyzed",
                "significant stimulation main effects, FDR < 0.05",
                "significant mating main effects, FDR < 0.05",
                "significant stimulation x mating interactions, FDR < 0.05",
                "exploratory interactions, P < 0.05 and abs(interaction log2FC) >= log2(1.2)",
            ],
            "value": [len(result), int((result.stimulation_FDR < 0.05).sum()), int((result.mating_FDR < 0.05).sum()), len(strict), len(exploratory)],
            "note": [
                "Two-factor OLS model on log2-transformed peak areas",
                "Averaged across AI and NM",
                "Averaged across US and ST",
                "Difference in stimulation response between NM and AI",
                "Candidate list for follow-up; not FDR-significant",
            ],
        }
    )

    result.to_csv(TABLE / "two_factor_model_all.csv", index=False, encoding="utf-8-sig")
    strict.to_csv(TABLE / "two_factor_significant_interaction.csv", index=False, encoding="utf-8-sig")
    exploratory.to_csv(TABLE / "two_factor_exploratory_interaction.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(TABLE / "two_factor_model_summary.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(TABLE / "two_factor_model_results.xlsx", engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name="all_metabolites", index=False)
            strict.to_excel(writer, sheet_name="FDR_interaction", index=False)
            exploratory.to_excel(writer, sheet_name="exploratory_interaction", index=False)
            summary.to_excel(writer, sheet_name="summary", index=False)
    except Exception as exc:
        print(f"Excel export skipped: {exc}")

    print(summary.to_string(index=False))
    print("Top interaction candidates:")
    print(result[["Compound_ID", "Name", "interaction_log2FC", "interaction_P", "interaction_FDR", "stim_log2FC_AI", "stim_log2FC_NM"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
