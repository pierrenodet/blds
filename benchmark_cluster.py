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
from bqlearn.baseline import BaselineBiqualityClassifier
from bqlearn.corruptions import make_imbalance
from bqlearn.reweighting import IRBL, KKMM, KPDR
from bqlearn.utils import safe_sparse_vstack
from pandas.core.frame import DataFrame
from scipy.stats import entropy
from sklearn import clone
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils import _safe_indexing
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import unique_labels

from covariate import IRBL2, PDR, make_cluster_imbalance, make_imbalance_index

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


seed = 42
base_clf = HistGradientBoostingClassifier(random_state=seed)


lr_classifiers = [
    (
        "irbl",
        IRBL(
            base_estimator=base_clf,
            final_estimator=base_clf,
        ),
    ),
    (
        "irbl2",
        IRBL2(
            base_estimator=base_clf,
            final_estimator=base_clf,
        ),
    ),
    (
        "trusted_only",
        BaselineBiqualityClassifier(estimator=base_clf, baseline="trusted_only"),
    ),
    (
        "no_correction",
        BaselineBiqualityClassifier(estimator=base_clf, baseline="no_correction"),
    ),
    (
        "kpdr",
        KPDR(base_estimator=base_clf, n_jobs=8, final_estimator=base_clf),
    ),
    (
        "pdr",
        PDR(base_estimator=base_clf, final_estimator=base_clf),
    ),
    (
        "kkmm",
        KKMM(kernel="rbf", batch_size=200, n_jobs=8, final_estimator=base_clf),
    ),
]


binary_datasets = [
    (
        "ad",
        fetch_openml(data_id=40978, return_X_y=True),
    ),
    ("eeg", fetch_openml(data_id=1471, return_X_y=True)),
    (
        "ibn_sina",
        (
            pd.read_csv("datasets/ibn-sina/train.csv", header=None, delimiter=" "),
            pd.read_csv("datasets/ibn-sina/label.csv", header=None, delimiter=" "),
        ),
    ),
    (
        "zebra",
        (
            pd.read_csv("datasets/zebra/train.csv", header=None, delimiter=" "),
            pd.read_csv("datasets/zebra/label.csv", header=None, delimiter=" "),
        ),
    ),
    ("musk", fetch_openml(data_id=1116, return_X_y=True)),
    (
        "phishing",
        fetch_openml(data_id=4534, return_X_y=True),
    ),
    ("spam", fetch_openml(data_id=44, return_X_y=True)),
    ("ijcnn1", fetch_openml(data_id=1575, return_X_y=True)),
    ("diabetes", fetch_openml(data_id=37, return_X_y=True)),
    (
        "credit-g",
        fetch_openml(data_id=31, return_X_y=True),
    ),
    (
        "hiva",
        (
            pd.read_csv("datasets/hiva/train.csv", header=None, delimiter=" "),
            pd.read_csv("datasets/hiva/label.csv", header=None, delimiter=" "),
        ),
    ),
    ("svmguide3", fetch_openml(data_id=1589, return_X_y=True)),
    ("web", fetch_openml(data_id=350, return_X_y=True)),
    (
        "mushroom",
        fetch_openml(data_id=24, return_X_y=True),
    ),
    ("skin-segmentation", fetch_openml(data_id=1502, return_X_y=True)),
    ("mozilla4", fetch_openml(data_id=1046, return_X_y=True)),
    ("electricity", fetch_openml(data_id=151, return_X_y=True)),
    (
        "bank-marketing",
        fetch_openml(data_id=1461, return_X_y=True),
    ),
    ("magic-telescope", fetch_openml(data_id=1120, return_X_y=True)),
    ("poker", fetch_openml(data_id=354, return_X_y=True)),
    ("phoeneme", fetch_openml(data_id=1489, return_X_y=True)),
]

multi_class_datasets = [
    ("ldpa", fetch_openml(data_id=1483, return_X_y=True)),
    ("letter", fetch_openml(data_id=6, return_X_y=True)),
    ("pendigits", fetch_openml(data_id=32, return_X_y=True)),
    ("har", fetch_openml(data_id=1478, return_X_y=True)),
    ("japanese-vowels", fetch_openml(data_id=375, return_X_y=True)),
    ("gas-drift", fetch_openml(data_id=1476, return_X_y=True)),
    ("mnist", fetch_openml(data_id=554, return_X_y=True)),
    (
        "covertype",
        fetch_openml(data_id=150, return_X_y=True),
    ),
    ("walking-activity", fetch_openml(data_id=1509, return_X_y=True)),
    (
        "connect-4",
        fetch_openml(data_id=40668, return_X_y=True),
    ),
    ("satimage", fetch_openml(data_id=182, return_X_y=True)),
    ("shuttle", fetch_openml(data_id=40685, return_X_y=True)),
    ("usps", fetch_openml(data_id=41082, return_X_y=True, as_frame=False)),
    (
        "dna",
        fetch_openml(data_id=40670, return_X_y=True),
    ),
    ("isolet", fetch_openml(data_id=300, return_X_y=True)),
    ("first-order-theorem-proving", fetch_openml(data_id=1475, return_X_y=True)),
    ("artificial-characters", fetch_openml(data_id=1459, return_X_y=True)),
    (
        "splice",
        fetch_openml(data_id=46, return_X_y=True),
    ),
    ("spoken-arabic-digits", fetch_openml(data_id=1503, return_X_y=True)),
]

parser = argparse.ArgumentParser()

parser.add_argument("output_dir", help="output directory name", type=str)

parser.add_argument(
    "--p",
    help="ratio of trusted data",
    nargs="+",
    type=float,
    default=[0.01, 0.02, 0.05],
)
parser.add_argument(
    "--subsampling",
    help="subsampling ratios",
    nargs="+",
    type=float,
    default=[1, 10, 20, 50],
)
parser.add_argument(
    "--cluster_subsampling",
    help="cluster subsampling ratios",
    nargs="+",
    type=float,
    default=[1, 20, 50, 100, 200, 500],
)
args = parser.parse_args()

ps = args.p
subsampling_ratios = args.subsampling
cluster_subsampling_ratios = args.cluster_subsampling

stats = []

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

for bin_ds_name, bin_ds in binary_datasets + multi_class_datasets:

    X, y = bin_ds

    if sp.issparse(X):
        X = X.todense()

    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.number):
        y = y.astype(int)
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    if isinstance(X, DataFrame):

        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

        continuous_features_selector = make_column_selector(dtype_include=np.number)
        categorical_features_selector = make_column_selector(
            dtype_include=[object, "category"]
        )

        ct = make_column_transformer(
            (
                make_pipeline(
                    StandardScaler(),
                    SimpleImputer(missing_values=np.nan, strategy="mean"),
                ),
                continuous_features_selector,
            ),
            (
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                categorical_features_selector,
            ),
        ).fit(X_train)

        X_train = ct.transform(X_train)
        X_test = ct.transform(X_test)

        n_categorical_features = len(categorical_features_selector(X))
        n_continous_features = len(continuous_features_selector(X))

    else:
        X_train = np.nan_to_num(X_train, nan=np.nan, neginf=np.nan, posinf=np.nan)
        X_test = np.nan_to_num(X_test, nan=np.nan, neginf=np.nan, posinf=np.nan)

        n_continous_features = n_features
        n_categorical_features = 0

    start = time.perf_counter()
    total = clone(base_clf).fit(X_train, y_train)
    end = time.perf_counter()

    y_pred = total.predict(X_test)
    y_score = total.decision_function(X_test)
    if y_score.ndim == 1:
        y_proba = np.c_[-y_score, y_score]
    else:
        y_proba = y_score
    softmax(y_proba, copy=False)
    y_proba = y_score

    total_acc = accuracy_score(y_test, y_pred)
    total_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
    total_kappa = cohen_kappa_score(y_test, y_pred)
    total_log_loss = log_loss(y_test, y_proba)

    stat = {
        "name": bin_ds_name,
        "n_features": n_features,
        "n_continous_features": n_continous_features,
        "n_categorical_features": n_categorical_features,
        "n_samples": n_samples,
        "n_classes": n_classes,
        "minority_class_ratio": np.min(np.bincount(y)) / n_samples,
        "normalized_entropy": entropy(np.bincount(y)) / math.log(n_classes),
        "total_acc": total_acc,
        "total_bacc": total_bacc,
        "total_kappa": total_kappa,
        "total_log_loss": total_log_loss,
        "fit_time": end - start,
    }

    with open(os.path.join(output_dir, "stats.csv"), "a+", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, stat.keys())
        if not os.path.getsize(os.path.join(output_dir, "stats.csv")):
            dict_writer.writeheader()
        dict_writer.writerow(stat)

    for p in ps:

        X_trusted, X_untrusted, y_trusted, y_untrusted = train_test_split(
            X_train, y_train, train_size=p, stratify=y_train, random_state=seed
        )

        list_k_means = make_cluster_imbalance(X_untrusted, y_untrusted)

        for cluster_subsampling_ratio in cluster_subsampling_ratios:

            Xs = []
            ys = []
            classes = unique_labels(y_untrusted)

            for i in range(n_classes):

                X_i = _safe_indexing(X_untrusted, y_untrusted == classes[i])
                y_i = _safe_indexing(y_untrusted, y_untrusted == classes[i])

                Xs.append(X_i)
                ys.append(y_i)

            Xs_imbalanced, ys_imbalanced = [], []

            for (X_i, y_i, k_means) in zip(Xs, ys, list_k_means):
                imbalanced_index = make_imbalance_index(
                    X_i,
                    k_means.predict(X_i),
                    majority_ratio=cluster_subsampling_ratio,
                    random_state=seed,
                )
                X_imbalanced = _safe_indexing(X_i, imbalanced_index)
                y_imbalanced = _safe_indexing(y_i, imbalanced_index)
                print(np.unique(k_means.predict(X_i), return_counts=True))
                print(X_imbalanced.shape, y_imbalanced.shape)
                Xs_imbalanced.append(X_imbalanced)
                ys_imbalanced.append(y_imbalanced)

            X_subsampled, y_subsampled = safe_sparse_vstack(Xs_imbalanced), np.hstack(
                ys_imbalanced
            )

            for subsampling_ratio in subsampling_ratios:

                X_sub_subsampled, y_sub_subsampled = make_imbalance(
                    X_subsampled,
                    y_subsampled,
                    majority_ratio=subsampling_ratio,
                    random_state=seed,
                )

                for clf_name, clf in lr_classifiers:

                    model = clf.fit_biquality(
                        X_trusted, y_trusted, X_sub_subsampled, y_sub_subsampled
                    )

                    y_pred = model.predict(X_test)
                    y_score = model.decision_function(X_test)
                    if y_score.ndim == 1:
                        y_proba = np.c_[-y_score, y_score]
                    else:
                        y_proba = y_score
                    softmax(y_proba, copy=False)
                    y_proba = y_score

                    acc = accuracy_score(y_test, y_pred)
                    bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                    kappa = cohen_kappa_score(y_test, y_pred)
                    logl = log_loss(y_test, y_proba)

                    res = {
                        "ds": bin_ds_name,
                        "p": p,
                        "cluster_subsampling_ratio": cluster_subsampling_ratio,
                        "subsampling_ratio": subsampling_ratio,
                        "acc": acc,
                        "bacc": bacc,
                        "kappa": kappa,
                        "log_loss": logl,
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
