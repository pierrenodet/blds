import argparse
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

from plot_utils import (
    error_curve_plot,
    nemenyi_plot,
    rank_curve_plot,
    score_to_rank,
    wilcoxon_plot,
    wilcoxon_test,
)
from table_utils import aggregated_results_to_latex, results_to_latex, stats_to_latex

matplotlib.use("Agg")


def do(input, stats, output_path, competitors, datasets, level, args):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    input = input[input["ds"].isin(datasets)]
    input = input[input["clf"].isin(competitors)]

    input_formatted = input.pivot_table(
        index=["ds", "perc_total_perf", level],
        columns="clf",
        values=args.perf,
        aggfunc="first",
    )

    levels = np.sort(input_formatted.index.unique(level=level).values)
    perc_total_perfs = np.sort(
        input_formatted.index.unique(level="perc_total_perf").values
    ).tolist()

    if args.nemenyi:
        avg_perfs = (
            input_formatted.groupby(["ds", "perc_total_perf"])
            .agg(
                lambda y: integrate.trapezoid(y, x=levels)
                / (levels.max() - levels.min())
            )
            .groupby("ds")
            .mean()
        )

        nemenyi_plot(
            len(datasets),
            zip(competitors.tolist(), avg_perfs.transpose().values.tolist()),
            reverse=True,
        )
        plt.savefig(
            os.path.join(output_path, f"nemenyi.pdf"),
            bbox_inches="tight",
            format="pdf",
        )
        plt.close()

        for perc_total_perf in perc_total_perfs:
            avg_perfs = (
                input_formatted.loc[(slice(None), perc_total_perf, slice(None)), :]
                .groupby("ds")
                .agg(lambda y: integrate.trapezoid(y, x=levels))
            )

            nemenyi_plot(
                len(datasets),
                zip(competitors.tolist(), avg_perfs.transpose().values.tolist()),
                reverse=True,
            )
            plt.savefig(
                os.path.join(output_path, f"nemenyi-{perc_total_perf}.pdf"),
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()

    input_table = round(
        input_formatted.groupby(["ds", "perc_total_perf"])
        .agg(
            [
                (
                    "mean",
                    lambda y: integrate.trapezoid(y, x=levels)
                    / (levels.max() - levels.min()),
                ),
                "std",
            ]
        )
        .groupby(["perc_total_perf"])
        .mean(),
        3,
    )

    with open(os.path.join(output_path, "table.tex"), "w") as tf:
        tf.write(aggregated_results_to_latex(input_table, args.perf, args.std))

    with open(os.path.join(output_path, "mean.tex"), "w") as tf:
        tf.write(round(input_table.mean(), 2).to_latex())

    if args.errcurve:
        output_directory_corruption_error_curve = os.path.join(
            output_path, "error_curves"
        )
        if not os.path.exists(output_directory_corruption_error_curve):
            os.mkdir(output_directory_corruption_error_curve)

        output_directory_corruption_rank_curve = os.path.join(
            output_path, "rank_curves"
        )
        if not os.path.exists(output_directory_corruption_rank_curve):
            os.mkdir(output_directory_corruption_rank_curve)

        for perc_total_perf in perc_total_perfs:
            error_curve_plot(
                zip(
                    competitors.tolist(),
                    input_formatted.loc[(slice(None), perc_total_perf, slice(None)), :]
                    .groupby(level)
                    .mean()
                    .transpose()
                    .values.tolist(),
                ),
                levels,
                ylabel=args.perf,
                xlabel=level,
            )
            plt.savefig(
                os.path.join(
                    output_path, "avg-error-curve-{}.pdf".format(perc_total_perf)
                ),
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()

            error_curve_plot(
                zip(
                    competitors.tolist(),
                    input_formatted.loc[(slice(None), perc_total_perf, slice(None)), :]
                    .rank(1, ascending=False, method="average")
                    .groupby(level)
                    .mean()
                    .transpose()
                    .values.tolist(),
                ),
                levels,
                ylabel="rank",
                xlabel=level,
            )
            plt.savefig(
                os.path.join(
                    output_path, "avg-rank-curve-{}.pdf".format(perc_total_perf)
                ),
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()

            rank_curve_plot(
                zip(
                    competitors.tolist(),
                    input_formatted.loc[(slice(None), perc_total_perf, slice(None)), :]
                    .groupby(level)
                    .mean()
                    .transpose()
                    .values.tolist(),
                ),
                levels,
                xlabel=level,
                reverse=True,
            )
            plt.savefig(
                os.path.join(
                    output_path,
                    "avg-error-curve-rankified-{}.pdf".format(perc_total_perf),
                ),
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()

            for dataset in datasets:
                error_curve_plot(
                    zip(
                        competitors.tolist(),
                        input_formatted.loc[(dataset, perc_total_perf, slice(None)), :]
                        .transpose()
                        .values.tolist(),
                    ),
                    levels,
                    args.perf,
                    xlabel=level,
                )
                plt.savefig(
                    os.path.join(
                        output_directory_corruption_error_curve,
                        "error-curve-{}-{}.pdf".format(dataset, perc_total_perf),
                    ),
                    bbox_inches="tight",
                    format="pdf",
                )
                plt.close()

                rank_curve_plot(
                    zip(
                        competitors.tolist(),
                        input_formatted.loc[(dataset, perc_total_perf, slice(None)), :]
                        .transpose()
                        .values.tolist(),
                    ),
                    levels,
                    xlabel=level,
                    reverse=True,
                )
                plt.savefig(
                    os.path.join(
                        output_directory_corruption_rank_curve,
                        "rank-curve-{}-{}.pdf".format(dataset, perc_total_perf),
                    ),
                    bbox_inches="tight",
                    format="pdf",
                )
                plt.close()

    if args.wilcoxon:
        perc_total_perfs = np.sort(
            input_formatted.index.unique(level="perc_total_perf").values
        )
        levels = np.sort(input_formatted.index.unique(level=level).values)
        for competitor1, competitor2 in itertools.product(competitors, competitors):
            if competitor1 == competitor2:
                continue
            zss = []
            for perc_total_perf in perc_total_perfs:
                zs = []
                for l in levels:
                    score1 = input_formatted.loc[
                        (slice(None), perc_total_perf, l), competitor1
                    ].values
                    score2 = input_formatted.loc[
                        (slice(None), perc_total_perf, l), competitor2
                    ].values
                    z, _ = wilcoxon_test(score1, score2, reverse=True)
                    zs.append(z)
                zss.append(zs)
            wilcoxon_plot(
                perc_total_perfs, levels, zss, xlabel=level, ylabel="perc_total_perf"
            )
            plt.savefig(
                os.path.join(
                    output_path,
                    "wilcoxon-{}-{}.pdf".format(competitor1, competitor2),
                ),
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="input file name")
    parser.add_argument("input_stats_file", help="input stats file name")
    parser.add_argument("output_directory", help="output directory name")

    parser.add_argument("--perf", help="performance metric", type=str, default="acc")
    parser.add_argument(
        "--wilcoxon", help="generate all wilcoxons", type=bool, default=True
    )
    parser.add_argument(
        "--errcurve", help="generate all error curves", type=bool, default=True
    )
    parser.add_argument(
        "--corr", help="generate correlation table", type=bool, default=True
    )
    parser.add_argument(
        "--nemenyi", help="generate nemenyi figures", type=bool, default=True
    )
    parser.add_argument(
        "--cross", help="generate cross-experiment figures", type=bool, default=True
    )
    parser.add_argument("--std", help="add std in tables", type=bool, default=False)

    args = parser.parse_args()

    input = pd.read_csv(args.input_file)
    input = input.drop_duplicates()
    input["r"] = round(1 - input["q"], 1)

    output_path = args.output_directory
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    competitors = np.sort(input["clf"].unique())
    datasets = np.sort(input["ds"].unique())

    stats = pd.read_csv(args.input_stats_file, index_col="name")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, "stats.tex"), "w") as tf:
        tf.write(
            stats_to_latex(
                stats[["n_samples", "n_features", "n_classes", "minority_class_ratio"]]
            )
        )

    with open(os.path.join(output_path, "results.tex"), "w") as tf:
        tf.write(results_to_latex(input, args.perf))

    for duo in [
        ["perc_total_perf", "p"],
        ["r", "noise_ratio"],
        ["cluster_imbalance_ratio", "data_ratio"],
    ]:
        with open(os.path.join(output_path, f"{duo[0]}.tex"), "w") as tf:
            tf.write(
                input[["ds", *duo]]
                .groupby(["ds", duo[0]])
                .first()
                .style.format(precision=3, subset=duo[1])
                .format_index(precision=2, level=duo[0])
                .to_latex(
                    hrules=True,
                    multirow_align="t",
                    multicol_align="c",
                )
            )

    no_imbalance = input[input["cluster_imbalance_ratio"] == 1.0]
    no_noise = input[input["r"] == 0.0]

    do(
        no_noise,
        stats,
        os.path.join(output_path, "no_noise"),
        competitors,
        datasets,
        "cluster_imbalance_ratio",
        args,
    )
    do(
        no_imbalance,
        stats,
        os.path.join(output_path, "no_imbalance"),
        competitors,
        datasets,
        "r",
        args,
    )

    if args.cross:
        cross_output_path = os.path.join(output_path, "cross")
        if not os.path.exists(cross_output_path):
            os.mkdir(cross_output_path)
        input_formatted = input.pivot_table(
            index=["ds", "perc_total_perf", "r", "cluster_imbalance_ratio"],
            columns="clf",
            values=args.perf,
            aggfunc="first",
        )

        if args.wilcoxon:
            perc_total_perfs = np.sort(
                input_formatted.index.unique(level="perc_total_perf").values
            )
            for perc_total_perf in perc_total_perfs:
                rs = np.sort(input_formatted.index.unique(level="r").values)
                cluster_imbalance_ratios = np.sort(
                    input_formatted.index.unique(level="cluster_imbalance_ratio").values
                )
                for competitor1, competitor2 in itertools.product(
                    competitors, competitors
                ):
                    if competitor1 == competitor2:
                        continue
                    zss = []
                    for r in rs:
                        zs = []
                        for cluster_imbalance_ratio in cluster_imbalance_ratios:
                            score1 = input_formatted.loc[
                                (
                                    slice(None),
                                    perc_total_perf,
                                    r,
                                    cluster_imbalance_ratio,
                                ),
                                competitor1,
                            ].values
                            score2 = input_formatted.loc[
                                (
                                    slice(None),
                                    perc_total_perf,
                                    r,
                                    cluster_imbalance_ratio,
                                ),
                                competitor2,
                            ].values
                            z, _ = wilcoxon_test(score1, score2, reverse=True)
                            zs.append(z)
                        zss.append(zs)
                    wilcoxon_plot(
                        rs,
                        cluster_imbalance_ratios,
                        zss,
                        xlabel="cluster_imbalance_ratio",
                        ylabel="r",
                    )
                    plt.savefig(
                        os.path.join(
                            cross_output_path,
                            "wilcoxon-{}-{}-{}.pdf".format(
                                competitor1, competitor2, perc_total_perf
                            ),
                        ),
                        bbox_inches="tight",
                        format="pdf",
                    )
                    plt.close()


if __name__ == "__main__":
    main()
