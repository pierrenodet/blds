import argparse
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import nemenyi_plot, wilcoxon_plot, wilcoxon_test


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
        "--corr", help="generate correlation table", type=bool, default=True
    )
    parser.add_argument(
        "--nemenyi", help="generate nemenyi figures", type=bool, default=True
    )

    args = parser.parse_args()

    input = pd.read_csv(args.input_file)
    input = input.drop_duplicates()

    output_path = args.output_directory
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    perf = args.perf

    competitors = np.sort(input["clf"].unique())
    datasets = np.sort(input["ds"].unique())

    input = input[
        (input["corruption"] == input["corruption"][0])
        & (input["subsampling_ratio"] != 500)
    ]
    
    stats = pd.read_csv(args.input_stats_file, index_col="name")

    stats["minority_class_ratio"] = round(stats["minority_class_ratio"], 2)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, "stats.tex"), "w") as tf:
        tf.write(
            stats[
                ["n_samples", "n_features", "n_classes", "minority_class_ratio"]
            ].to_latex()
        )

    input_formatted = input.pivot_table(
        index=["ds", "q", "p", "subsampling_ratio"],
        columns="clf",
        values=perf,
        aggfunc="first",
    )
    output_directory_overall = os.path.join(output_path, "overall")
    if not os.path.exists(output_directory_overall):
        os.mkdir(output_directory_overall)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    subsampling_ratios = np.sort(
        input_formatted.index.unique(level="subsampling_ratio").values
    )

    if args.nemenyi:

        avg_perfs = input_formatted.groupby("ds").mean()
        
        nemenyi_plot(
            len(datasets),
            zip(competitors.tolist(), avg_perfs.transpose().values.tolist()),
            reverse=True,
        )
        plt.savefig(
            os.path.join(output_directory_overall, f"nemenyi.pdf"),
            bbox_inches="tight",
            format="pdf",
        )

    if args.wilcoxon:

        ps = np.sort(input_formatted.index.unique(level="p").values)
        qs = np.sort(input_formatted.index.unique(level="q").values)
        for p in ps:

            for competitor1, competitor2 in itertools.product(competitors, competitors):
                if competitor1 == competitor2:
                    continue
                zss = []
                for m in subsampling_ratios:
                    zs = []
                    for q in qs:
                        score1 = input_formatted.loc[
                            (slice(None), q, p, m), competitor1
                        ].values
                        score2 = input_formatted.loc[
                            (slice(None), q, p, m), competitor2
                        ].values
                        z, _ = wilcoxon_test(score1, score2, reverse=True)
                        zs.append(z)
                    zs.reverse()
                    zss.append(zs)
                wilcoxon_plot(subsampling_ratios, np.flip(np.round(1 - qs,1)), zss, ylabel="ϱ", xlabel="r")
                plt.savefig(
                    os.path.join(
                        output_directory_overall,
                        "wilcoxon-{}-{}-{}.pdf".format(competitor1, competitor2, p),
                    ),
                    bbox_inches="tight",
                    format="pdf",
                )

    # No noise
    no_noise = input[(input["q"] == 1) & (input["corruption"] == "uniform")]
    no_noise = no_noise.drop(["q", "corruption"], axis=1)
    no_noise = no_noise.drop_duplicates()

    no_noise_formatted = no_noise.pivot_table(
        index=["ds", "p", "subsampling_ratio"],
        columns="clf",
        values=perf,
        aggfunc="first",
    )

    no_noise_table = round(
        no_noise_formatted.groupby(["p", "subsampling_ratio"]).agg(["mean", "std"])
        * 100,
        2,
    )

    print(no_noise_table)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, "no_noise.tex"), "w") as tf:
        tf.write(no_noise_table.to_latex())

    avg_perfs = no_noise_formatted.groupby("ds").mean()
    output_directory_covariate = os.path.join(args.output_directory, "covariate")
    if not os.path.exists(output_directory_covariate):
        os.mkdir(output_directory_covariate)

    if args.nemenyi:

        avg_perfs = (
            no_noise_formatted.loc[(slice(None), slice(None), slice(None)), :]
            .groupby("ds")
            .mean()
        )
        nemenyi_plot(
            len(datasets),
            zip(competitors.tolist(), avg_perfs.transpose().values.tolist()),
            reverse=True,
        )
        plt.savefig(
            os.path.join(output_directory_covariate, f"nemenyi.pdf"),
            bbox_inches="tight",
            format="pdf",
        )

        for subsampling_ratio in subsampling_ratios:
            avg_perfs = (
                no_noise_formatted.loc[(slice(None), slice(None), subsampling_ratio), :]
                .groupby("ds")
                .mean()
            )
            nemenyi_plot(
                len(datasets),
                zip(competitors.tolist(), avg_perfs.transpose().values.tolist()),
                reverse=True,
            )
            plt.savefig(
                os.path.join(
                    output_directory_covariate, f"nemenyi-{subsampling_ratio}.pdf"
                ),
                bbox_inches="tight",
                format="pdf",
            )

    if args.wilcoxon:
        ps = np.sort(no_noise_formatted.index.unique(level="p").values)
        for competitor1, competitor2 in itertools.product(competitors, competitors):
            if competitor1 == competitor2:
                continue
            zss = []
            for p in ps:
                zs = []
                for m in subsampling_ratios:
                    score1 = no_noise_formatted.loc[
                        (slice(None), p, m), competitor1
                    ].values
                    score2 = no_noise_formatted.loc[
                        (slice(None), p, m), competitor2
                    ].values
                    z, _ = wilcoxon_test(score1, score2, reverse=True)
                    zs.append(z)
                zss.append(zs)
            qs = subsampling_ratios
            wilcoxon_plot(ps, qs, zss)
            plt.savefig(
                os.path.join(
                    output_directory_covariate,
                    "wilcoxon-{}-{}.pdf".format(competitor1, competitor2),
                ),
                bbox_inches="tight",
                format="pdf",
            )

    # No covariate
    no_covariate = input[input["subsampling_ratio"] == 1]
    no_covariate = no_covariate.drop(["subsampling_ratio"], axis=1)
    no_covariate = no_covariate.drop_duplicates()

    no_covariate_formatted = no_covariate.pivot_table(
        index=["ds", "p", "q", "corruption"],
        columns="clf",
        values=perf,
        aggfunc="first",
    )

    no_covariate_table = round(
        no_covariate_formatted.groupby(["ds", "corruption", "p"])
        .agg(["mean", "std"])
        .groupby(["corruption", "p"])
        .mean()
        * 100,
        2,
    )
    print(no_covariate_table)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(os.path.join(output_path, "no_covariate.tex"), "w") as tf:
        tf.write(no_covariate_table.to_latex())

    for corruption in no_covariate_formatted.index.unique(level="corruption").values:

        no_covariate_per_corruption = no_covariate_formatted.filter(
            like=corruption, axis=0
        )

        output_directory_corruption = os.path.join(args.output_directory, corruption)
        if not os.path.exists(output_directory_corruption):
            os.mkdir(output_directory_corruption)

        avg_perfs = no_covariate_per_corruption.groupby("ds").mean()

        if args.corr:
            stats_names = stats.columns
            corrs = avg_perfs.join(stats).corr()[stats_names].drop(stats_names, axis=0)

            with open(
                os.path.join(output_directory_corruption, "corrs.tex"), "w"
            ) as tf:
                tf.write(corrs.to_latex())

            with open(
                os.path.join(output_directory_corruption, "stats.tex"), "w"
            ) as tf:
                tf.write(stats.to_latex())

        if args.nemenyi:
            nemenyi_plot(
                len(datasets),
                zip(competitors.tolist(), avg_perfs.transpose().values.tolist()),
                reverse=True,
            )
            plt.savefig(
                os.path.join(output_directory_corruption, "nemenyi.pdf"),
                bbox_inches="tight",
                format="pdf",
            )

        if args.wilcoxon:
            ps = np.sort(no_covariate_per_corruption.index.unique(level="p").values)
            qs = np.sort(no_covariate_per_corruption.index.unique(level="q").values)
            for competitor1, competitor2 in itertools.product(competitors, competitors):
                if competitor1 == competitor2:
                    continue
                zss = []
                for p in ps:
                    zs = []
                    for q in qs:
                        score1 = no_covariate_per_corruption.loc[
                            (slice(None), p, q), competitor1
                        ].values
                        score2 = no_covariate_per_corruption.loc[
                            (slice(None), p, q), competitor2
                        ].values
                        z, _ = wilcoxon_test(score1, score2, reverse=True)
                        zs.append(z)
                    zss.append(zs)
                wilcoxon_plot(ps, qs, zss)
                plt.savefig(
                    os.path.join(
                        output_directory_corruption,
                        "wilcoxon-{}-{}.pdf".format(competitor1, competitor2),
                    ),
                    bbox_inches="tight",
                    format="pdf",
                )


if __name__ == "__main__":
    main()
