import math

import matplotlib.pyplot as plt
import numpy as np
import Orange
from scipy import stats


def score_to_rank(*args, reverse=False):
    k = len(args)
    if k < 2:
        raise ValueError("Less than 2 levels")
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1:
        raise ValueError("Unequal number of samples")

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row, reverse=reverse)
        rankings.append(
            [int(row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2) for v in row]
        )

    return list(map(list, zip(*rankings)))


def friedman_test(*args, reverse=False):
    k = len(args)
    if k < 2:
        raise ValueError("Less than 2 levels")
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1:
        raise ValueError("Unequal number of samples")

    rankings = score_to_rank(*args, reverse=reverse)
    rankings = list(map(list, zip(*rankings)))

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6.0 * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
        (np.sum([r**2 for r in rankings_avg])) - ((k * (k + 1) ** 2) / float(4))
    )
    iman_davenport = ((n - 1) * chi2) / float((n * (k - 1) - chi2))

    p_value = 1 - stats.f.cdf(iman_davenport, k - 1, (k - 1) * (n - 1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def wilcoxon_test(score_A, score_B, reverse=False):
    # compute abs delta and sign
    if reverse:
        delta_score = score_A - score_B
    else:
        delta_score = score_B - score_A
    sign_delta_score = np.sign(delta_score)
    abs_delta_score = np.abs(delta_score)

    N_r = len(delta_score)

    # sort
    sorted_indexes = np.argsort(abs_delta_score)
    sorted_sign_delta_score = sign_delta_score[sorted_indexes]
    ranks = np.arange(1, N_r + 1)

    # z : pouput value
    W = np.sum(sorted_sign_delta_score * ranks)
    z = W / (math.sqrt(N_r * (N_r + 1) * (2 * N_r + 1) / 6.0))

    # rejecte or not the null hypothesis
    null_hypothesis_rejected = False
    if z < -1.96 or z > 1.96:
        null_hypothesis_rejected = True

    return z, null_hypothesis_rejected


def nemenyi_plot(n, args, reverse=False):
    plt.clf()
    names, ranks = map(list, zip(*args))
    names = [clf_to_pclf[name] for name in names]
    avranks = friedman_test(*ranks, reverse=reverse)[2]
    cd = Orange.evaluation.compute_CD(avranks, n)
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1)


def wilcoxon_plot(ps, qs, zs, xlabel="q", ylabel="p"):
    plt.clf()

    plt.figure(figsize=(8, 4))

    xlabel = level_to_plevel[xlabel]
    ylabel = level_to_plevel[ylabel]

    cps = np.linspace(0, 1, len(ps))
    cqs = np.linspace(0, 1, len(qs))

    cpss, cqss = np.meshgrid(cps, cqs, indexing="ij")

    zs = np.array(zs)

    z_ceil = 1.96

    wins = zs > z_ceil
    losses = zs < -z_ceil
    ties = (zs < z_ceil) & (zs > -z_ceil)

    plt.scatter(cqss[wins], cpss[wins], color="black", facecolor="white", label="win")
    plt.scatter(cqss[ties], cpss[ties], color="black", marker=".", s=1, label="tie")
    plt.scatter(cqss[losses], cpss[losses], color="black", label="loss")

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    plt.xticks(ticks=cqs, labels=qs, fontsize=14)
    plt.yticks(ticks=cps, labels=ps, fontsize=14)

    plt.tight_layout()


def error_curve_plot(args, qs, ylabel="acc", ymin=None, ymax=None, xlabel="q"):
    plt.clf()

    xlabel = level_to_plevel[xlabel]
    ylabel = metric_to_pmetric[ylabel]

    plt.figure(figsize=(6, 4))

    for name, perfs in args:
        name = clf_to_pclf[name]
        plt.plot(qs, perfs, label=name)

    plt.legend(loc="upper right")

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # plt.ylim((ymin, ymax))

    plt.tight_layout()


from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator


def rank_curve_plot(args, qs, xlabel="q", reverse=False):
    plt.clf()

    xlabel = level_to_plevel[xlabel]

    names, scores = map(list, zip(*args))
    n = len(names)
    names = [clf_to_pclf[name] for name in names]
    ranks = score_to_rank(*scores, reverse=reverse)
    last_ranks = list(map(list, zip(*ranks)))[-1]

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(ylim=(0.5, 0.5 + n)))

    ax.xaxis.set_ticks(qs)
    ax.yaxis.set_major_locator(MultipleLocator(1))

    yax2 = ax.secondary_yaxis("right")
    yax2.yaxis.set_major_locator(FixedLocator(range(1, n + 1)))
    yax2.yaxis.set_major_formatter(
        FixedFormatter([name for _, name in sorted(zip(last_ranks, names))])
    )

    for rank in ranks:
        ax.plot(qs, rank, "o-", mfc="w")

    ax.invert_yaxis()
    ax.set(xlabel=xlabel, ylabel="Rank")
    ax.grid(axis="x")

    plt.tight_layout()


level_to_plevel = {
    "r": "r",
    "cluster_imbalance_ratio": "$\\rho$",
    "perc_total_perf": "p",
    "ds": "Dataset",
}

metric_to_pmetric = {
    "kappa": "$\\kappa$",
    "acc": "accuracy",
    "bacc": "balanced accuracy",
    "log_loss": "log loss",
    "rank": "rank",
}

clf_to_pclf = {
    "irbl": "IRBL",
    "irbl2": "IRBL2",
    "kkmm": "$K$-KMM",
    "kmm": "KMM",
    "kpdr": "$K$-PDR",
    "pdr": "PDR",
    "no_correction": "No Correction",
    "trusted_only": "Trusted Only",
    "untrusted_only": "Untrusted Only",
}
