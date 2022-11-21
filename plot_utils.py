import math

import matplotlib.pyplot as plt
import numpy as np
import Orange
from scipy import stats


def friedman_test(*args, reverse=False):

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
            [row_sort.index(v) + 1 + (row_sort.count(v) - 1) / 2.0 for v in row]
        )

    rankings_avg = [np.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (k + 1) / (6.0 * n)) for r in rankings_avg]

    chi2 = ((12 * n) / float((k * (k + 1)))) * (
        (np.sum([r ** 2 for r in rankings_avg])) - ((k * (k + 1) ** 2) / float(4))
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
    avranks = friedman_test(*ranks, reverse=reverse)[2]
    cd = Orange.evaluation.compute_CD(avranks, n)
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1)


def wilcoxon_plot(ps, qs, zs, xlabel="p", ylabel="cluster_subsampling_ratio"):

    plt.clf()

    plt.figure(figsize=(6, 3.5))

    cps = np.linspace(0, 1, len(ps))
    cqs = np.linspace(0, 1, len(qs))

    cpss, cqss = np.meshgrid(cps, cqs, indexing="ij")

    zs = np.array(zs)

    z_ceil = 2.02

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
