from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse


RANDOM_SEED = 20260629
N_PERMUTATIONS = 200
N_FOLDS = 7

US_AI_COLOR = "#98A8A4"
ST_AI_COLOR = "#F1D68F"
TEXT_COLOR = "#000000"


def project_dirs():
    base = Path(__file__).resolve().parents[1]
    return {
        "base": base,
        "input": base / "输入文件",
        "output": base / "输出文件",
        "figures": base / "图片",
    }


def read_table(path):
    return pd.read_csv(path, sep="\t")


def load_data(input_dir):
    raw = read_table(input_dir / "US_AI.vs.ST_AI_all.xls")
    sample_cols = [c for c in raw.columns if str(c).startswith("all_")]
    keep = [c for c in sample_cols if c.startswith("all_US_AI_") or c.startswith("all_ST_AI_")]
    if len(keep) != 14:
        raise RuntimeError(f"Expected 14 US_AI/ST_AI sample columns, found {len(keep)}")

    x_raw = raw[keep].apply(pd.to_numeric, errors="coerce").T
    x_raw.index = [s.replace("all_", "") for s in keep]
    x_raw.columns = raw["Compound_ID"].astype(str).tolist()
    y = np.array([1 if s.startswith("US_AI_") else 0 for s in x_raw.index], dtype=float)
    group = np.array(["US_AI" if v == 1 else "ST_AI" for v in y])

    meta = pd.DataFrame(
        {
            "Sample": x_raw.index,
            "Group": group,
            "Vendor_ID_num": [
                int(s.split("_")[-1]) if s.startswith("US_AI_") else int(s.split("_")[-1]) + 13
                for s in x_raw.index
            ],
        }
    )
    compound_info = raw[
        [
            "Compound_ID",
            "Name",
            "Formula",
            "RT[min]",
            "CAS No.",
            "Avg Calculated RI",
            "Library RI",
        ]
    ].copy()
    return x_raw, y, group, meta, compound_info


def preprocess(x_raw, mode="sqrt_autoscale"):
    values = x_raw.values.astype(float)
    if mode == "raw_autoscale":
        x = values.copy()
    elif mode == "log10_autoscale":
        x = np.log10(values + 1.0)
    elif mode == "sqrt_autoscale":
        x = np.sqrt(values)
    else:
        raise ValueError(mode)

    center = x.mean(axis=0)
    scale = x.std(axis=0, ddof=1)
    scale[scale == 0] = 1.0
    return (x - center) / scale


def pls1_fit(x, y, n_components=2):
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=1)
    x_std[x_std == 0] = 1.0
    y_mean = y.mean()

    xh = (x - x_mean) / x_std
    yh = y.astype(float) - y_mean

    weights = []
    loadings = []
    y_loadings = []
    scores = []
    x_explained = []
    ssx_total = float(np.sum(xh * xh))

    for _ in range(n_components):
        w = xh.T @ yh
        norm = np.linalg.norm(w)
        if norm < 1e-12:
            break
        w = w / norm
        t = xh @ w
        denom = float(t.T @ t)
        if denom < 1e-12:
            break
        p = xh.T @ t / denom
        q = float(yh.T @ t / denom)
        x_hat = np.outer(t, p)
        x_explained.append(float(np.sum(x_hat * x_hat) / ssx_total))
        xh = xh - x_hat
        yh = yh - q * t
        weights.append(w)
        loadings.append(p)
        y_loadings.append(q)
        scores.append(t)

    if not weights:
        coef = np.zeros(x.shape[1])
        score_matrix = np.zeros((x.shape[0], n_components))
        weights_matrix = np.zeros((x.shape[1], 0))
        loadings_matrix = np.zeros((x.shape[1], 0))
    else:
        weights_matrix = np.column_stack(weights)
        loadings_matrix = np.column_stack(loadings)
        q = np.asarray(y_loadings).reshape(-1, 1)
        coef_scaled = weights_matrix @ np.linalg.pinv(loadings_matrix.T @ weights_matrix) @ q
        coef = coef_scaled.ravel() / x_std
        score_matrix = np.column_stack(scores)
        if score_matrix.shape[1] < n_components:
            pad = np.zeros((x.shape[0], n_components - score_matrix.shape[1]))
            score_matrix = np.column_stack([score_matrix, pad])

    fitted = x @ coef + (y_mean - x_mean @ coef)
    sse = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2y = 1.0 - sse / sst if sst > 0 else np.nan

    return {
        "coef": coef,
        "intercept": y_mean - x_mean @ coef,
        "scores": score_matrix[:, :n_components],
        "weights": weights_matrix,
        "loadings": loadings_matrix,
        "x_explained": np.asarray(x_explained),
        "r2y": r2y,
        "fitted": fitted,
    }


def predict(model, x):
    return x @ model["coef"] + model["intercept"]


def make_folds(y, n_folds=7, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    folds = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[i] = k % n_folds
    return folds


def cross_validated_q2(x, y, n_components=2, n_folds=7, seed=RANDOM_SEED):
    folds = make_folds(y, n_folds=n_folds, seed=seed)
    pred = np.zeros(len(y), dtype=float)
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        model = pls1_fit(x[train], y[train], n_components=n_components)
        pred[test] = predict(model, x[test])
    press = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    q2 = 1.0 - press / sst if sst > 0 else np.nan
    cls = (pred >= 0.5).astype(float)
    accuracy = float(np.mean(cls == y))
    return q2, accuracy, pred, folds


def permute_validation(x, y, n_components=2, n_permutations=200, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    observed = pls1_fit(x, y, n_components=n_components)
    obs_q2, obs_acc, obs_pred, folds = cross_validated_q2(
        x, y, n_components=n_components, n_folds=N_FOLDS, seed=seed
    )
    y_centered = y - y.mean()
    rows = [
        {
            "cor": 1.0,
            "R2": observed["r2y"],
            "Q2": obs_q2,
            "kind": "observed",
        }
    ]
    for i in range(n_permutations):
        yp = rng.permutation(y)
        model = pls1_fit(x, yp, n_components=n_components)
        q2, _, _, _ = cross_validated_q2(
            x, yp, n_components=n_components, n_folds=N_FOLDS, seed=seed + i + 1
        )
        cor = float(np.corrcoef(y_centered, yp - yp.mean())[0, 1])
        if not np.isfinite(cor):
            cor = 0.0
        cor = abs(cor)
        rows.append({"cor": cor, "R2": model["r2y"], "Q2": q2, "kind": "permuted"})
    perm = pd.DataFrame(rows)
    r2_fit = np.polyfit(perm["cor"], perm["R2"], 1)
    q2_fit = np.polyfit(perm["cor"], perm["Q2"], 1)
    intercepts = {
        "R2_intercept_at_cor0": float(np.polyval(r2_fit, 0.0)),
        "Q2_intercept_at_cor0": float(np.polyval(q2_fit, 0.0)),
        "R2_at_cor1_fit": float(np.polyval(r2_fit, 1.0)),
        "Q2_at_cor1_fit": float(np.polyval(q2_fit, 1.0)),
        "R2Y_observed": float(observed["r2y"]),
        "Q2Y_observed": float(obs_q2),
        "CV_accuracy_observed": float(obs_acc),
    }
    return observed, obs_q2, obs_acc, obs_pred, folds, perm, intercepts, r2_fit, q2_fit


def confidence_ellipse(ax, x, y, color):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(vals * 5.991)
    ell = Ellipse(
        (np.mean(x), np.mean(y)),
        width,
        height,
        angle=angle,
        fill=False,
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(ell)


def setup_style():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.edgecolor"] = TEXT_COLOR
    plt.rcParams["axes.labelcolor"] = TEXT_COLOR
    plt.rcParams["xtick.color"] = TEXT_COLOR
    plt.rcParams["ytick.color"] = TEXT_COLOR
    plt.rcParams["text.color"] = TEXT_COLOR
    plt.rcParams["axes.grid"] = False


def format_p_value(p_value):
    if p_value < 0.0001:
        return "< 0.0001"
    return f"{p_value:.4f}"


def load_score_permanova(output_dir):
    path = output_dir / "US_AI_vs_ST_AI_PERMANOVA_results.csv"
    if not path.exists():
        return None
    results = pd.read_csv(path)
    hit = results[
        (results["analysis"] == "PLS-DA score plot")
        & (results["input_space"] == "PC1 and PC2 scores from reproduced PLS-DA")
    ]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return float(row["R2"]), float(row["P_permutation"])


def save_score_plot(figures_dir, output_dir, model, meta, intercepts):
    scores = model["scores"].copy()
    x_exp = model["x_explained"]
    if len(x_exp) < 2:
        x_exp = np.pad(x_exp, (0, 2 - len(x_exp)), constant_values=np.nan)

    plot_df = meta.copy()
    plot_df["PC1"] = scores[:, 0]
    plot_df["PC2"] = scores[:, 1]

    fig, ax = plt.subplots(figsize=(8.9 / 2.54, 8.9 / 2.54))
    group_styles = {
        "US_AI": {"color": US_AI_COLOR, "marker": "^", "label": "US.AI"},
        "ST_AI": {"color": ST_AI_COLOR, "marker": "s", "label": "ST.AI"},
    }
    for grp in ["US_AI", "ST_AI"]:
        style = group_styles[grp]
        sub = plot_df[plot_df["Group"] == grp]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=30,
            color=style["color"],
            marker=style["marker"],
            edgecolor=TEXT_COLOR,
            linewidth=0.5,
            zorder=3,
        )
        confidence_ellipse(ax, sub["PC1"].values, sub["PC2"].values, style["color"])

    ax.axhline(0, color="#BFBFBF", linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.axvline(0, color="#BFBFBF", linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xlabel(f"PC1 ({x_exp[0] * 100:.2f}%)", fontsize=9.4)
    ax.set_ylabel(f"PC2 ({x_exp[1] * 100:.2f}%)", fontsize=9.4)
    permanova_result = load_score_permanova(output_dir)
    if permanova_result is not None:
        permanova_r2, permanova_p = permanova_result
        ax.text(
            0.04,
            0.96,
            "PERMANOVA\n"
            + f"$R^2$ = {permanova_r2:.3f}, $\\it{{P}}$ = {format_p_value(permanova_p)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            color=TEXT_COLOR,
        )
    ax.text(
        0.5,
        -0.18,
        f"R2Y: {intercepts['R2Y_observed']:.2f}, Q2Y: {intercepts['Q2Y_observed']:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.4,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=group_styles["US_AI"]["marker"],
            color="none",
            markerfacecolor=US_AI_COLOR,
            markeredgecolor=TEXT_COLOR,
            label=group_styles["US_AI"]["label"],
            markersize=5,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker=group_styles["ST_AI"]["marker"],
            color="none",
            markerfacecolor=ST_AI_COLOR,
            markeredgecolor=TEXT_COLOR,
            label=group_styles["ST_AI"]["label"],
            markersize=5,
            linestyle="None",
        ),
    ]
    ax.legend(
        handles=handles,
        title=None,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=8.2, width=0.7, length=3)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.22, top=0.82)
    for ext in ["pdf", "jpg"]:
        path = figures_dir / f"US_AI_vs_ST_AI_PLSDA_score_from_raw.{ext}"
        if ext == "jpg":
            fig.savefig(path, dpi=600)
        else:
            fig.savefig(path)
    plt.close(fig)


def save_validation_plot(figures_dir, perm, intercepts, r2_fit, q2_fit):
    fig, ax = plt.subplots(figsize=(8.9 / 2.54, 8.9 / 2.54))
    x_line = np.linspace(0, 1, 100)
    ax.scatter(perm["cor"], perm["R2"], s=18, color=US_AI_COLOR, marker="o", label="R2")
    ax.scatter(perm["cor"], perm["Q2"], s=18, color=ST_AI_COLOR, marker="s", label="Q2")
    ax.plot(x_line, np.polyval(r2_fit, x_line), color=US_AI_COLOR, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.plot(x_line, np.polyval(q2_fit, x_line), color=ST_AI_COLOR, linewidth=0.9, linestyle=(0, (3, 3)))
    ax.set_xlabel("Cor", fontsize=9.4)
    ax.set_ylabel("Value", fontsize=9.4)
    title = (
        "US_AI_ST_AI\n"
        f"Intercepts:R2=(0.0,{intercepts['R2_intercept_at_cor0']:.2f}), "
        f"Q2=(0.0,{intercepts['Q2_intercept_at_cor0']:.2f})"
    )
    ax.set_title(title, fontsize=9.4, fontweight="bold", pad=6)
    ax.legend(loc="lower right", frameon=True, edgecolor=TEXT_COLOR, fancybox=False, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(TEXT_COLOR)
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=8.2, width=0.7, length=3)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.16, top=0.82)
    for ext in ["pdf", "jpg"]:
        path = figures_dir / f"US_AI_vs_ST_AI_PLSDA_validation_from_raw.{ext}"
        if ext == "jpg":
            fig.savefig(path, dpi=600)
        else:
            fig.savefig(path)
    plt.close(fig)


def main():
    dirs = project_dirs()
    dirs["output"].mkdir(parents=True, exist_ok=True)
    dirs["figures"].mkdir(parents=True, exist_ok=True)
    setup_style()

    x_raw, y, group, meta, compound_info = load_data(dirs["input"])
    x = preprocess(x_raw, mode="sqrt_autoscale")
    (
        model,
        q2,
        cv_acc,
        cv_pred,
        folds,
        perm,
        intercepts,
        r2_fit,
        q2_fit,
    ) = permute_validation(x, y, n_components=2, n_permutations=N_PERMUTATIONS)

    score_df = meta.copy()
    score_df["PLSDA_PC1"] = model["scores"][:, 0]
    score_df["PLSDA_PC2"] = model["scores"][:, 1]
    score_df["CV_fold"] = folds
    score_df["CV_predicted_US_AI_score"] = cv_pred
    score_df["CV_predicted_group"] = np.where(cv_pred >= 0.5, "US_AI", "ST_AI")
    score_df.to_csv(dirs["output"] / "US_AI_vs_ST_AI_PLSDA_scores_from_raw.csv", index=False, encoding="utf-8-sig")

    coef_df = pd.DataFrame({"Compound_ID": x_raw.columns, "PLSDA_coefficient": model["coef"]})
    coef_df = coef_df.merge(compound_info, on="Compound_ID", how="left")
    coef_df["abs_coefficient"] = coef_df["PLSDA_coefficient"].abs()
    coef_df.sort_values("abs_coefficient", ascending=False).to_csv(
        dirs["output"] / "US_AI_vs_ST_AI_PLSDA_coefficients_from_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    perm.to_csv(dirs["output"] / "US_AI_vs_ST_AI_PLSDA_permutation_validation_from_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([intercepts]).to_csv(
        dirs["output"] / "US_AI_vs_ST_AI_PLSDA_summary_from_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "input_file": "US_AI.vs.ST_AI_all.xls",
                "comparison": "US_AI vs ST_AI",
                "preprocessing_used_for_main_figures": "sqrt(raw intensity), mean-centered and autoscaled per metabolite",
                "plsda_components": 2,
                "cross_validation": "stratified 7-fold",
                "permutations": N_PERMUTATIONS,
                "note": "This is an open-code reproduction from the vendor intensity table. Vendor software and exact random CV partitions are unknown, so numerical values may not be identical.",
            }
        ]
    ).to_csv(dirs["output"] / "US_AI_vs_ST_AI_PLSDA_reproduction_notes.csv", index=False, encoding="utf-8-sig")

    save_score_plot(dirs["figures"], dirs["output"], model, meta, intercepts)
    save_validation_plot(dirs["figures"], perm, intercepts, r2_fit, q2_fit)

    print("Output base:", dirs["base"])
    print(pd.DataFrame([intercepts]).to_string(index=False))
    print("X explained:", model["x_explained"])


if __name__ == "__main__":
    main()
