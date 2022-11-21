import warnings
from functools import partial

import numpy as np
from bqlearn.reweighting import IRBL
from bqlearn.reweighting._base import BaseReweightingBiqualityClassifier
from bqlearn.reweighting._kmm import kmm
from bqlearn.utils import safe_sparse_vstack
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.utils import (_safe_indexing, check_consistent_length,
                           check_random_state, check_scalar, gen_batches)
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import _num_samples


def make_cluster_imbalance(
    X,
    y,
    *,
    n_clusters=[2, 3, 4],
    random_state=None,
    n_jobs=None,
):
    """
    Create class imbalance in a multi class scenario according to [1]_.

    It selects randomly some classes to be considered as the minority group
    according to `minority_class_fraction` and going to subsample it given
    `majority_ratio` when `imbalance_distribution='step'`.

    If `imbalance_distribution='linear'`, it creates imbalance between all classes by
    decreasing linearly the ratio of subsampling when iterating through classes
    according to `majority_ratio`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    y : array-like of shape (n_samples, )
        The targets.

    majority_ratio : float, default = 1.0
        Ratio between number of samples in majority classes and
        number of samples in minority classes.

    imbalance_distribution : {'step', 'linear'}, default='step'
        Imbalance distribution.

    minority_class_fraction : float, default = 0.5
        Fraction of classes considered as minority classes. Only used
        when `imbalance_distribution='step'`.

    Returns
    -------
    X_imbalanced : array-like of shape (n_samples_new, n_features)
        The array containing the imbalanced data.

    y_imbalanced : array-like of shape (n_samples_new)
        The corresponding label of X_imbalanced.

    References
    ----------
    .. [1] Mateusz Buda, et al. "A systematic study of the class imbalance problem
        in convolutional neural networks." Neural Networks, 106:249–259, 2018.
    """

    random_state = check_random_state(random_state)

    check_consistent_length(X, y)

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    classes = unique_labels(y)
    n_classes = len(classes)

    def optimized_k_means_silouhette(X, list_n_clusters):
        list_k_means = [
            MiniBatchKMeans(n_clusters=n_cluster, random_state=random_state)
            for n_cluster in list_n_clusters
        ]

        def fit_predict(k_means, X):
            return k_means.fit_predict(X)

        list_cluster_labels = Parallel(n_jobs=n_jobs)(
            delayed(partial(fit_predict, X=X))(k_means) for k_means in list_k_means
        )
        silhouette_scores = [
            silhouette_score(X, cluster_labels)
            for cluster_labels in list_cluster_labels
        ]
        best_k = np.argmax(silhouette_scores)
        print(silhouette_scores)
        return list_k_means[best_k]

    Xs = []

    for i in range(n_classes):

        X_i = _safe_indexing(X, y == classes[i])

        Xs.append(X_i)

    best_k_means = Parallel(n_jobs=n_jobs)(
        delayed(optimized_k_means_silouhette)(X_i, n_clusters) for X_i in Xs
    )

    return best_k_means


def make_imbalance_index(
    X,
    y,
    *,
    majority_ratio=1.0,
    imbalance_distribution="step",
    minority_class_fraction=0.5,
    random_state=None,
):
    """
    Create class imbalance in a multi class scenario according to [1]_.

    It selects randomly some classes to be considered as the minority group
    according to `minority_class_fraction` and going to subsample it given
    `majority_ratio` when `imbalance_distribution='step'`.

    If `imbalance_distribution='linear'`, it creates imbalance between all classes by
    decreasing linearly the ratio of subsampling when iterating through classes
    according to `majority_ratio`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    y : array-like of shape (n_samples, )
        The targets.

    majority_ratio : float, default = 1.0
        Ratio between number of samples in majority classes and
        number of samples in minority classes.

    imbalance_distribution : {'step', 'linear'}, default='step'
        Imbalance distribution.

    minority_class_fraction : float, default = 0.5
        Fraction of classes considered as minority classes. Only used
        when `imbalance_distribution='step'`.

    Returns
    -------
    X_imbalanced : array-like of shape (n_samples_new, n_features)
        The array containing the imbalanced data.

    y_imbalanced : array-like of shape (n_samples_new)
        The corresponding label of X_imbalanced.

    References
    ----------
    .. [1] Mateusz Buda, et al. "A systematic study of the class imbalance problem
        in convolutional neural networks." Neural Networks, 106:249–259, 2018.
    """

    random_state = check_random_state(random_state)

    check_consistent_length(X, y)

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    classes = unique_labels(y)
    n_classes = len(classes)

    check_scalar(
        majority_ratio,
        "majority_ratio",
        (float, int),
        min_val=1,
        include_boundaries="left",
    )
    check_scalar(
        minority_class_fraction,
        "minority_class_fraction",
        (float, int),
        min_val=0,
        max_val=1,
        include_boundaries="neither",
    )

    n_samples_per_class = np.bincount(y)
    idx_minority_classes = np.argsort(n_samples_per_class)[::-1]

    if imbalance_distribution == "linear":
        sampling_probability = np.linspace(1 / majority_ratio, 1.0, n_classes)
        sampling_probability = sampling_probability[idx_minority_classes]

    elif imbalance_distribution == "step":
        sampling_probability = np.ones(n_classes)
        n_minority_classes = round(n_classes * minority_class_fraction)
        sampling_probability[idx_minority_classes[:n_minority_classes]] = (
            1 / majority_ratio
        )
    else:
        raise ValueError(
            f"Unsupported imbalance distribution : {imbalance_distribution}"
        )

    acc = np.empty((0,), dtype=int)

    for i, c in enumerate(classes):
        idx = random_state.choice(
            range(np.count_nonzero(y == c)),
            size=round(n_samples_per_class[i] * sampling_probability[i]),
            replace=False,
        )

        acc = np.concatenate(
            (
                acc,
                np.flatnonzero(y == c)[idx],
            ),
            axis=0,
        )

    return acc


class IRBL2(IRBL):
    """An improved IRBL.

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the IRBL is built.
        Support for probability prediction is required.

    final_estimator : object, optional (default=None)
        The final estimator from which the IRBL is built.
        Support for sample weighting is required.

    random_state : int or RandomState, default=None
        Controls the random seed given at base_estimator and final_estimator.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    estimator_ : classifier
        The final fitted estimator.

    beta_ : ndarray, shape (n_samples,)
        The weights of the examples computed during :meth:`fit`.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.
    """

    def __init__(self, base_estimator=None, final_estimator=None, random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            final_estimator=final_estimator,
            random_state=random_state,
        )

    def _reweight(self, X_trusted, X_untrusted, y_trusted, y_untrusted):

        beta_trusted, beta_untrusted = super()._reweight(
            X_trusted, X_untrusted, y_trusted, y_untrusted
        )

        if self.base_estimator is None:
            self.estimator_shift_ = LogisticRegression()
        else:
            self.estimator_shift_ = clone(self.base_estimator)

        if hasattr(self.estimator_shift_, "random_state"):
            self.estimator_shift_.set_params(random_state=self.random_state)

        if not hasattr(self.estimator_shift_.__class__, "predict_proba"):
            raise ValueError(
                """%s doesn't support predict_proba."""
                % self.base_estimator.__class__.__name__
            )

        n_samples_trusted = _num_samples(X_trusted)
        n_samples_untrusted = _num_samples(X_untrusted)

        self.estimator_shift_.fit(
            safe_sparse_vstack((X_trusted, X_untrusted)),
            np.hstack((np.ones(n_samples_trusted), np.zeros(n_samples_untrusted))),
        )

        prior = n_samples_untrusted / n_samples_trusted

        if isinstance(self.estimator_shift_, LogisticRegression):
            beta = np.exp(self.estimator_shift_.decision_function(X_untrusted))
        else:
            proba = self.estimator_shift_.predict_proba(X_untrusted)
            beta = 1 / proba[:, 0] - 1

        return beta_trusted, beta_untrusted * beta * prior


class KMM(BaseReweightingBiqualityClassifier):
    """A KMM Biquality Classifier

    Parameters
    ----------
    kernel : str or callable, default="rbf"
        Kernel mapping used internally. This parameter is directly passed to
        :class:`~sklearn.metrics.pairwise.pairwise_kernel`.
        If `kernel` is a string, it must be one of the metrics
        in `pairwise.PAIRWISE_KERNEL_FUNCTIONS`.
        If `kernel` is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if `kernel` is a callable function, it is called on
        each pair of instances (rows) and the resulting value recorded. The
        callable should take two rows from X as input and return the
        corresponding kernel value as a single number. This means that
        callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
        they operate on matrices, not single samples. Use the string
        identifying the kernel instead.

    kernel_params : dict, optional (default={})
        Kernel additional parameters

    B: float, optional (default=1000)
        Bounding weights parameter.

    epsilon: float, optional (default=None)
        Constraint parameter.
        If ``None`` epsilon is set to
        ``(np.sqrt(n_samples_untrusted - 1)/np.sqrt(n_samples_untrusted)``.

    max_iter : int, default=100
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations.

    batch_size : int, default='auto'
        Size of minibatches for batched Kernel Mean Matching.
        When set to "auto", ``batch_size=int(0.05*n_samples_untrusted)``.
        When set to None, ``batch_size=n_samples_untrusted``.

    final_estimator : object, optional (default=None)
        The final estimator from which the KMMDensityRatioClassifier is built.
        Support for sample weighting is required.

    n_jobs : int, default=None
        The number of jobs to use for the computation.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int or RandomState, default=None
        Controls the random seed given at final_estimator.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    estimator_ : classifier
        The final fitted estimator.

    beta_ : ndarray, shape (n_samples,)
        The weights of the examples computed during :meth:`fit`.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    References
    ----------
    .. [1] Miao Y., Farahat A. and Kamel M.
        "Ensemble Kernel Mean Matching", 2015
    .. [2] Huang, J. and Smola, A. and Gretton, A. and Borgwardt, KM. and Schölkopf, B.,
        "Correcting Sample Selection Bias by Unlabeled Data", 2006
    """

    def __init__(
        self,
        kernel="rbf",
        kernel_params={},
        B=1000,
        epsilon=None,
        max_iter=100,
        batch_size="auto",
        final_estimator=None,
        n_jobs=None,
        random_state=None,
    ):

        super().__init__(final_estimator=final_estimator, random_state=random_state)

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.B = B
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def _reweight(self, X_trusted, X_untrusted, y_trusted, y_untrusted):

        n_samples_trusted = _num_samples(X_trusted)
        n_samples_untrusted = _num_samples(X_untrusted)

        if self.batch_size == "auto":
            batch_size = int(0.05 * n_samples_untrusted)
        elif self.batch_size is None:
            batch_size = n_samples_untrusted
        else:
            if self.batch_size < 1 or self.batch_size > n_samples_untrusted:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped."
                )
            batch_size = np.clip(self.batch_size, 1, n_samples_untrusted)

        batch_slices = gen_batches(n_samples_untrusted, batch_size)

        kmms = Parallel(n_jobs=self.n_jobs)(
            delayed(kmm)(
                X_trusted,
                X_untrusted[batch_slice],
                kernel=self.kernel,
                kernel_params=self.kernel_params,
                B=self.B,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
            )
            for batch_slice in batch_slices
        )

        return np.ones(n_samples_trusted), np.concatenate(kmms)


class PDR(BaseReweightingBiqualityClassifier):
    """A PDR Classifier

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the IRBL is built.
        Support for probability prediction is required.

    final_estimator : object, optional (default=None)
        The final estimator from which the IRBL is built.
        Support for sample weighting is required.

    random_state : int or RandomState, default=None
        Controls the random seed given at base_estimator and final_estimator.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    estimator_ : classifier
        The final fitted estimator.

    beta_ : ndarray, shape (n_samples,)
        The weights of the examples computed during :meth:`fit`.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.
    """

    def __init__(self, base_estimator=None, final_estimator=None, random_state=None):

        super().__init__(
            final_estimator=final_estimator,
            random_state=random_state,
        )

        self.base_estimator = base_estimator

    def _reweight(self, X_trusted, X_untrusted, y_trusted, y_untrusted):

        if self.base_estimator is None:
            self.estimator_shift_ = LogisticRegression()
        else:
            self.estimator_shift_ = clone(self.base_estimator)

        if hasattr(self.estimator_shift_, "random_state"):
            self.estimator_shift_.set_params(random_state=self.random_state)

        if not hasattr(self.estimator_shift_.__class__, "predict_proba"):
            raise ValueError(
                """%s doesn't support predict_proba."""
                % self.base_estimator.__class__.__name__
            )

        n_samples_trusted = _num_samples(X_trusted)
        n_samples_untrusted = _num_samples(X_untrusted)

        self.estimator_shift_.fit(
            safe_sparse_vstack((X_trusted, X_untrusted)),
            np.hstack([np.ones(n_samples_trusted), np.zeros(n_samples_untrusted)]),
        )

        prior = n_samples_untrusted / n_samples_trusted

        if isinstance(self.estimator_shift_, LogisticRegression):
            beta = np.exp(self.estimator_shift_.decision_function(X_untrusted))
        else:
            proba = self.estimator_shift_.predict_proba(X_untrusted)
            beta = 1 / proba[:, 0] - 1

        return np.ones(n_samples_trusted), beta * prior