import pandas as pd
from plot_utils import clf_to_pclf, metric_to_pmetric, level_to_plevel


def rename_classifiers(df):
    return df.rename(
        columns=clf_to_pclf,
    )


def results_to_latex(results, perf):
    results_formatted = results.pivot_table(
        index=["ds", "perc_total_perf", "cluster_imbalance_ratio", "r"],
        columns="clf",
        values=perf,
        aggfunc="first",
    )

    return (
        rename_classifiers(results_formatted)
        .rename_axis(
            index=level_to_plevel
        )
        .rename_axis("", axis=1)
        .style.format(precision=4)
        .format_index(precision=1, level="r")
        .format_index(precision=2, level="p")
        .to_latex(
            hrules=True,
            multirow_align="t",
            multicol_align="c",
            environment="longtable",
            caption=f"Table of results for {perf} metric.",
        )
    )


def aggregated_results_to_latex(agg_results, perf):
    formatted_results = pd.DataFrame(
        columns=agg_results.columns.get_level_values(0).unique(),
        index=agg_results.index.get_level_values(0),
    )
    for col in agg_results.columns.get_level_values(0).unique():
        formatted_results.loc[:, col] = (
            agg_results.loc[:, (col, "mean")]
            .astype(str)
            .str.cat(agg_results.loc[:, (col, "std")].astype(str), sep=" $\\pm$ ")
        )

    return (
        rename_classifiers(formatted_results)
        .reset_index(names="p")
        .rename_axis(
            index=level_to_plevel
        )
        .style.format(precision=2, subset="p")
        .hide(axis=0)
        .to_latex(
            hrules=True,
            multirow_align="t",
            multicol_align="c",
            caption=f"Table of results for {perf} metric.",
        )
    )


def safe_num(num):
    if isinstance(num, str):
        num = float(num)
    return float("{:.3g}".format(abs(num)))


def format_number(num):
    num = safe_num(num)
    sign = ""

    metric = {"T": 1000000000000, "B": 1000000000, "M": 1000000, "K": 1000, "": 1}

    for index in metric:
        num_check = num / metric[index]

        if num_check >= 1:
            num = num_check
            sign = index
            break

    return f"{str(num).rstrip('0').rstrip('.')}{sign}"


def stats_to_latex(stats):
    stats["n_samples"] = stats["n_samples"].apply(format_number)
    return (
        stats.reset_index(names="Datasets")
        .rename(
            columns={
                "n_samples": "$ \\vert D \\vert $",
                "n_features": "$ \\vert \\mathcal{X} \\vert $",
                "n_classes": "$ \\vert \\mathcal{Y} \\vert $",
                "minority_class_ratio": "min",
            }
        )
        .style.format(precision=2, subset=["min"])
        .hide(axis=0)
        .to_latex(
            label="datasets",
            hrules=True,
            multirow_align="t",
            multicol_align="c",
            caption="""Multi-class classification datasets used for the evaluation. Columns: number of samples ($ \\vert D \\vert $), number of features ($ \\vert \\mathcal{X} \\vert $), number of classes ($ \\vert \\mathcal{Y} \\vert $), and ratio of the number of samples from the minority class over the number of samples from the majority class (min).""",
        )
    )
