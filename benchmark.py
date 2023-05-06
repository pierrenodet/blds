import argparse
import csv
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from bqlearn.baseline import make_baseline
from bqlearn.corruptions import make_imbalance, make_instance_dependent_label_noise
from bqlearn.irbl import IRBL
from bqlearn.kdr import KKMM, KPDR
from pandas.core.frame import DataFrame
from scipy import interpolate
from scipy.stats import entropy
from sklearn import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    log_loss,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from covariate import IRBL2, KMM, PDR, find_best_kmeans, noisy_leaves_probability

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


seed = 42
base_clf = HistGradientBoostingClassifier(early_stopping=False, random_state=seed)

cv = 5

lr_classifiers = [
    (
        "irbl",
        IRBL(
            base_estimator=CalibratedClassifierCV(
                base_clf, method="isotonic", n_jobs=-1, cv=cv
            ),
            final_estimator=base_clf,
        ),
    ),
    (
        "irbl2",
        IRBL2(
            base_estimator=CalibratedClassifierCV(
                base_clf, method="isotonic", n_jobs=-1, cv=cv
            ),
            covariate_estimator=base_clf,
            final_estimator=base_clf,
        ),
    ),
    (
        "trusted_only",
        make_baseline(estimator=base_clf, baseline="trusted_only"),
    ),
    (
        "no_correction",
        make_baseline(estimator=base_clf, baseline="no_correction"),
    ),
    (
        "untrusted_only",
        make_baseline(estimator=base_clf, baseline="untrusted_only"),
    ),
    (
        "kpdr",
        KPDR(
            base_estimator=base_clf,
            n_jobs=-1,
            final_estimator=base_clf,
        ),
    ),
    (
        "pdr",
        PDR(
            base_estimator=base_clf,
            final_estimator=base_clf,
        ),
    ),
    (
        "kkmm",
        KKMM(
            final_estimator=base_clf,
            kernel="rbf",
            batch_size=100,
            max_iter=10000,
            tol=1e-6,
            n_jobs=-1,
        ),
    ),
    (
        "kmm",
        KMM(
            final_estimator=base_clf,
            kernel="rbf",
            batch_size=100,
            max_iter=10000,
            tol=1e-6,
            n_jobs=-1,
        ),
    ),
]

binary_datasets = [
    ("eeg", fetch_openml(data_id=1471, return_X_y=True, parser="pandas")),
    # (
    #     "zebra",
    #     (
    #         pd.read_csv("datasets/zebra/train.csv", header=None, delimiter=" "),
    #         pd.read_csv("datasets/zebra/label.csv", header=None, delimiter=" "),
    #     ),
    # ),
    # ("musk", fetch_openml(data_id=1116, return_X_y=True, parser="pandas")),
    # ("phishing", fetch_openml(data_id=4534, return_X_y=True, parser="pandas")),
    # ("spam", fetch_openml(data_id=44, return_X_y=True, parser="pandas")),
    # ("ijcnn1", fetch_openml(data_id=1575, return_X_y=True)),
    # ("diabetes", fetch_openml(data_id=37, return_X_y=True, parser="pandas")),
    # ("credit-g", fetch_openml(data_id=31, return_X_y=True, parser="pandas")),
    # ("svmguide3", fetch_openml(data_id=1589, return_X_y=True)),
    # ("web", fetch_openml(data_id=350, return_X_y=True)),
    # ("mushroom", fetch_openml(data_id=24, return_X_y=True, parser="pandas")),
    # ("skin-segmentation", fetch_openml(data_id=1502, return_X_y=True, parser="pandas")),
    # ("mozilla4", fetch_openml(data_id=1046, return_X_y=True, parser="pandas")),
    # ("electricity", fetch_openml(data_id=151, return_X_y=True, parser="pandas")),
    # ("bank-marketing", fetch_openml(data_id=1461, return_X_y=True, parser="pandas")),
    # ("magic-telescope", fetch_openml(data_id=1120, return_X_y=True, parser="pandas")),
    # ("phoeneme", fetch_openml(data_id=1489, return_X_y=True, parser="pandas")),
    # ("nomao", fetch_openml(data_id=1486, return_X_y=True, parser="pandas")),
    # ("click", fetch_openml(data_id=1220, return_X_y=True, parser="pandas")),
    # ("jm1", fetch_openml(data_id=1053, return_X_y=True, parser="pandas")),
    # ("poker", fetch_openml(data_id=354, return_X_y=True)),
    # (
    #     "hiva",
    #     (
    #         pd.read_csv("datasets/hiva/train.csv", header=None, delimiter=" "),
    #         pd.read_csv("datasets/hiva/label.csv", header=None, delimiter=" "),
    #     ),
    # ),
    # (
    #     "ibn-sina",
    #     (
    #         pd.read_csv("datasets/ibn-sina/train.csv", header=None, delimiter=" "),
    #         pd.read_csv("datasets/ibn-sina/label.csv", header=None, delimiter=" "),
    #     ),
    # ),
    # ("ad", fetch_openml(data_id=40978, return_X_y=True, parser="pandas")),
]

multi_class_datasets = [
    # ("pendigits", fetch_openml(data_id=32, return_X_y=True, parser="pandas")),
    # ("har", fetch_openml(data_id=1478, return_X_y=True, parser="pandas")),
    # ("japanese-vowels", fetch_openml(data_id=375, return_X_y=True, parser="pandas")),
    # ("gas-drift", fetch_openml(data_id=1476, return_X_y=True, parser="pandas")),
    # ("walking-activity", fetch_openml(data_id=1509, return_X_y=True, parser="pandas")),
    # ("connect-4", fetch_openml(data_id=40668, return_X_y=True, parser="pandas")),
    # ("satimage", fetch_openml(data_id=182, return_X_y=True, parser="pandas")),
    # (
    #     "usps",
    #     fetch_openml(data_id=41082, return_X_y=True, as_frame=False, parser="pandas"),
    # ),
    # ("dna", fetch_openml(data_id=40670, return_X_y=True, parser="pandas")),
    # ("isolet", fetch_openml(data_id=300, return_X_y=True, parser="pandas")),
    # ("ldpa", fetch_openml(data_id=1483, return_X_y=True, parser="pandas")),
    # ("mnist", fetch_openml(data_id=554, return_X_y=True, parser="pandas")),
    # ("covertype", fetch_openml(data_id=150, return_X_y=True, parser="pandas")),
    # ("letter", fetch_openml(data_id=6, return_X_y=True, parser="pandas")),
]

parser = argparse.ArgumentParser()

parser.add_argument("output_dir", help="output directory name", type=str)

parser.add_argument(
    "--perc_total_perf",
    help="ratio of trusted data given as a percentage of max performance",
    nargs="+",
    type=float,
    default=[0.4, 0.6, 0.75],
)
parser.add_argument(
    "--q",
    help="corruption strength",
    nargs="+",
    type=float,
    default=[1.0],
)
parser.add_argument(
    "--corruption",
    help="corruption kind",
    nargs="+",
    type=str,
    default=["permutation"],
)
# parser.add_argument(
#     "--cluster_imbalance",
#     help="cluster imbalance ratios",
#     nargs="+",
#     type=float,
#     default=[1, 2, 5, 10, 20, 50, 100],
# )
parser.add_argument(
    "--cluster_split",
    help="cluster split ratios",
    nargs="+",
    type=float,
    default=[0.0, 0.3, 0.6, 1.0],
)
args = parser.parse_args()

perc_total_perfs = args.perc_total_perf
qs = args.q[::-1]
# cluster_imbalance_ratios = args.cluster_imbalance
cluster_splits = args.cluster_split
corruptions = args.corruption

stats = []

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

for bin_ds_name, bin_ds in binary_datasets + multi_class_datasets:
    X, y = bin_ds

    if sp.issparse(X):
        X = X.toarray()

    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
        y = y.astype(int)
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # X_train, X_test, y_train, y_test, train, test = train_test_split(
    #     X, y, range(n_samples), test_size=0.3, stratify=y, random_state=seed
    # )

    if isinstance(X, DataFrame):
        X = X.replace([np.inf, -np.inf], np.nan)
        # X_train = X_train.replace([np.inf, -np.inf], np.nan)
        # X_test = X_test.replace([np.inf, -np.inf], np.nan)

        continuous_features_selector = make_column_selector(dtype_include=np.number)
        categorical_features_selector = make_column_selector(
            dtype_include=[object, "category"]
        )

        ct = make_column_transformer(
            (
                make_pipeline(
                    SimpleImputer(missing_values=np.nan, strategy="mean"),
                ),
                continuous_features_selector,
            ),
            (
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                categorical_features_selector,
            ),
        ).fit(X)

        n_categorical_features = len(categorical_features_selector(X))
        n_continous_features = len(continuous_features_selector(X))

        X = ct.transform(X)
        # X_train = ct.transform(X_train)
        # X_test = ct.transform(X_test)

    else:
        X = np.nan_to_num(X, nan=np.nan, neginf=np.nan, posinf=np.nan)
        # X_train = np.nan_to_num(X_train, nan=np.nan, neginf=np.nan, posinf=np.nan)
        # X_test = np.nan_to_num(X_test, nan=np.nan, neginf=np.nan, posinf=np.nan)

        n_continous_features = n_features
        n_categorical_features = 0

    # start = time.perf_counter()
    # total = clone(base_clf).fit(X_train, y_train)
    # end = time.perf_counter()

    # y_pred = total.predict(X_test)
    # y_proba = total.predict_proba(X_test)

    # total_acc = accuracy_score(y_test, y_pred)
    # total_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
    # total_kappa = cohen_kappa_score(y_test, y_pred)
    # total_log_loss = log_loss(y_test, y_proba)

    kmeans = find_best_kmeans(X, y, n_clusters=[2], random_state=seed)

    # stat = {
    #     "name": bin_ds_name,
    #     "n_features": n_features,
    #     "n_continous_features": n_continous_features,
    #     "n_categorical_features": n_categorical_features,
    #     "n_samples": n_samples,
    #     "n_classes": n_classes,
    #     "minority_class_ratio": np.min(np.bincount(y)) / np.max(np.bincount(y)),
    #     "normalized_entropy": entropy(np.bincount(y)) / math.log(n_classes),
    #     "n_clusters": list(map(lambda k_means: k_means.n_clusters, kmeans)),
    #     "total_acc": total_acc,
    #     "total_bacc": total_bacc,
    #     "total_kappa": total_kappa,
    #     "total_log_loss": total_log_loss,
    #     "fit_time": end - start,
    # }

    # with open(os.path.join(output_dir, "stats.csv"), "a+", newline="") as output_file:
    #     dict_writer = csv.DictWriter(output_file, stat.keys())
    #     if not os.path.getsize(os.path.join(output_dir, "stats.csv")):
    #         dict_writer.writeheader()
    #     dict_writer.writerow(stat)

    tree = DecisionTreeClassifier(min_samples_leaf=0.1 / n_classes, random_state=seed)
    tree.fit(X, y)

    train_sizes = np.array(
        [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
    )
    _, _, test_scores = learning_curve(
        base_clf,
        X,
        y,
        train_sizes=train_sizes,
        random_state=seed,
        scoring=make_scorer(cohen_kappa_score),
        cv=10,
        n_jobs=-1,
    )
    test_scores = test_scores.mean(axis=1)

    test_scores = test_scores.ravel()
    n_test_scores = len(test_scores)
    print(f"test scores: {test_scores}")
    max_perf = np.max(test_scores)
    p = interpolate.interp1d(
        test_scores, train_sizes[-n_test_scores:], fill_value="extrapolate"
    )

    ps = [p(perc_total_perf * max_perf).item() for perc_total_perf in perc_total_perfs]

    print(f"trusted data ratios: {ps}")

    for p, perc_total_perf in zip(ps, perc_total_perfs):
        for cluster_split in cluster_splits:
            trusted = []
            untrusted = []
            for i in range(n_classes):
                mask = y == i
                n_samples_i = mask.sum()
                clusters = kmeans[i].predict(X[mask])
                n_clusters = kmeans[i].n_clusters
                sizes = np.bincount(clusters, minlength=n_clusters)
                print(sizes)
                probas = (
                    (clusters == 1) * (cluster_split)
                    + (1 - cluster_split) / n_clusters
                )
                trusted.append(
                    np.random.choice(
                        np.arange(n_samples)[mask],
                        size=int(0.3 * n_samples_i) + int(p * n_samples_i),
                        replace=True,
                        p=probas / probas.sum(),
                    )
                )
                untrusted.append(
                    np.random.choice(
                        np.arange(n_samples)[mask],
                        size=int(0.7 * n_samples_i),
                        replace=True,
                        p=(1 - probas) / (1 - probas).sum(),
                    )
                )

            trusted = np.concatenate(trusted)
            untrusted = np.concatenate(untrusted)

            trusted, test = next(
                StratifiedShuffleSplit(
                    n_splits=1, train_size=int(p * n_samples), random_state=seed
                ).split(X[trusted], y[trusted])
            )

            X_test, y_test = X[test], y[test]
            X_train, y_train = (
                X[np.hstack((trusted, untrusted))],
                y[np.hstack((trusted, untrusted))],
            )

            print(
                X_train.shape,
                X_test.shape,
                trusted.shape[0],
                untrusted.shape[0],
                n_samples,
            )
            sample_quality = np.hstack(
                (np.ones(trusted.shape[0]), np.zeros(untrusted.shape[0]))
            )

            for q in qs:
                noise_prob = noisy_leaves_probability(
                    X,
                    tree,
                    noise_ratio=1 - q,
                    random=False,
                    ascending=True,
                    random_state=seed,
                )

                for corruption in corruptions:
                    y_corrupted = np.copy(y)

                    y_corrupted[untrusted] = make_instance_dependent_label_noise(
                        noise_prob[untrusted],
                        y[untrusted],
                        corruption,
                        random_state=seed,
                        labels=classes,
                    )

                    for clf_name, clf in lr_classifiers:
                        model = clf.fit(
                            X[np.hstack((trusted, untrusted))],
                            y_corrupted[np.hstack((trusted, untrusted))],
                            sample_quality=np.hstack(
                                (
                                    np.ones(trusted.shape[0]),
                                    np.zeros(untrusted.shape[0]),
                                )
                            ),
                        )

                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)

                        acc = accuracy_score(y_test, y_pred)
                        bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                        kappa = cohen_kappa_score(y_test, y_pred)
                        logl = log_loss(y_test, y_proba)

                        res = {
                            "ds": bin_ds_name,
                            "p": round(p, 4),
                            "perc_total_perf": perc_total_perf,
                            "q": q,
                            # "cluster_imbalance_ratio": cluster_imbalance_ratio,
                            "cluster_split": cluster_split,
                            # "data_ratio": round(data_ratio, 4),
                            # "noise_ratio": round(noise_ratio, 2),
                            "acc": round(acc, 4),
                            "bacc": round(bacc, 4),
                            "kappa": round(kappa, 4),
                            "log_loss": round(logl, 4),
                            "clf": clf_name,
                        }
                        print(res)

                        with open(
                            os.path.join(output_dir, "results.csv"), "a+", newline=""
                        ) as output_file:
                            dict_writer = csv.DictWriter(output_file, res.keys())
                            if not os.path.getsize(
                                os.path.join(output_dir, "results.csv")
                            ):
                                dict_writer.writeheader()
                            dict_writer.writerow(res)
