import warnings
from abc import ABCMeta, abstractmethod
from functools import partial

import numpy as np
from bqlearn.baseline import make_baseline
from bqlearn.irbl import IRBL
from bqlearn.kdr import kmm, pdr
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils import (
    _safe_indexing,
    check_array,
    check_consistent_length,
    check_random_state,
    gen_batches,
    safe_mask,
    shuffle,
)
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
    unique_labels,
)
from sklearn.utils.validation import _num_samples, check_is_fitted, has_fit_parameter


def noisy_leaves_probability(
    X,
    tree,
    *,
    noise_ratio=0.5,
    random=True,
    ascending=True,
    random_state=None,
):
    """Noisify some leaves of a decision tree learn on the input dataset.
    These leaves can be chosen completly at random or prioritizing the most pure leaves.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The samples.

    tree : DecisionTree

    noise_ratio : float, default=0.5
        The ratio of noise. Must be between 0 and 1.

    random: boolean, default=True
        Choose leaves completly at random or prioritize pure leaves.

    random_state : int or RandomState, default=None
        Controls the training of the DecisionTreeClassifier and
        the noisy leaves selection.

    Returns
    -------
    noise_probabilities : array-like of shape (n_samples, )
        The noise probabilities.
    """

    X_leaves = tree.apply(X)
    leaves = np.unique(X_leaves)

    leaves = shuffle(leaves, random_state=random_state)

    if random:
        sorted_leaves = leaves
    else:
        impurity = tree.tree_.impurity[leaves]
        if not ascending:
            impurity = - impurity
        sorted_leaves = leaves[np.argsort(impurity)]

    n_samples = _num_samples(X)
    n_selected_leaves = (
        np.cumsum(tree.tree_.n_node_samples[sorted_leaves]) >= noise_ratio * n_samples
    ).argmax()
    selected_leaves = sorted_leaves[:n_selected_leaves]

    return np.isin(X_leaves, selected_leaves).astype(float)


def find_best_kmeans(
    X,
    y,
    *,
    n_clusters=[8],
    random_state=None,
    n_jobs=None,
):
    check_consistent_length(X, y)

    y_type = type_of_target(y)
    if y_type not in ("binary", "multiclass"):
        raise ValueError("%s is not supported" % y_type)

    classes = unique_labels(y)
    n_classes = len(classes)

    def optimized_k_means_silouhette(X, list_n_clusters):
        list_k_means = [
            MiniBatchKMeans(
                batch_size=256 * 32, n_clusters=n_cluster, random_state=random_state
            )
            for n_cluster in list_n_clusters
        ]

        def fit_predict(k_means, X):
            try:
                res = k_means.fit_predict(X)
            except:
                res = np.random.randint(0, _num_samples(X), size=X.shape[0])
            return res

        list_cluster_labels = Parallel(n_jobs=n_jobs)(
            delayed(partial(fit_predict, X=X))(k_means) for k_means in list_k_means
        )
        silhouette_scores = [
            silhouette_score(X, cluster_labels)
            for cluster_labels in list_cluster_labels
        ]
        best_k = np.argmax(silhouette_scores)

        print(f"silouhette scores: {silhouette_scores}")
        return list_k_means[best_k]

    Xs = []

    for i in range(n_classes):
        X_i = _safe_indexing(X, y == classes[i])

        Xs.append(X_i)

    best_k_means = Parallel(n_jobs=n_jobs)(
        delayed(optimized_k_means_silouhette)(X_i, n_clusters) for X_i in Xs
    )

    return best_k_means


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

    def __init__(self, base_estimator, covariate_estimator, final_estimator):
        self.base_estimator = base_estimator
        self.covariate_estimator = covariate_estimator
        self.final_estimator = final_estimator

    def fit(self, X, y, sample_quality=None):
        """Fit the reweighted model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_quality : array-like, shape (n_samples,)
            Sample qualities.

        Returns
        -------
        self : object
        """

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )

        check_classification_targets(y)

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_classes_ = len(self.classes_)

        y = self._le.transform(y)

        if sample_quality is not None:
            sample_quality = check_array(
                sample_quality, input_name="sample_quality", ensure_2d=False
            )
        else:
            raise ValueError("The 'sample_quality' parameter should not be None.")

        if not hasattr(self.base_estimator.__class__, "predict_proba"):
            raise ValueError(
                """%s doesn't support predict_proba."""
                % self.base_estimator.__class__.__name__
            )

        if not has_fit_parameter(self.final_estimator, "sample_weight"):
            raise ValueError(
                "%s doesn't support sample_weight."
                % self.final_estimator.__class__.__name__
            )

        self._estimator_trusted = make_baseline(self.base_estimator, "trusted_only")
        self._estimator_untrusted = make_baseline(self.base_estimator, "untrusted_only")

        self._estimator_trusted.fit(X, y, sample_quality=sample_quality)
        self._estimator_untrusted.fit(X, y, sample_quality=sample_quality)

        n_samples = _num_samples(X)
        Y = label_binarize(y, classes=range(self.n_classes_))
        if Y.shape[1] == 1:
            Y = np.hstack((1 - Y, Y))

        y_prob_trusted = self._estimator_trusted.predict_proba(X)
        y_prob_untrusted = self._estimator_untrusted.predict_proba(X)

        num = np.sum(y_prob_trusted * Y, axis=1)
        den = np.sum(y_prob_untrusted * Y, axis=1)

        self.sample_weight_ = np.divide(
            num,
            den,
            out=np.zeros(n_samples),
            where=den != 0,
        )
        self.sample_weight_[sample_quality == 1] = 1
        self.sample_weight_[sample_quality == 0] *= pdr(
            X[sample_quality == 0], X[sample_quality == 1], self.covariate_estimator
        )

        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(X, y, sample_weight=self.sample_weight_)

        return self


class BaseReweightingBiqualityClassifier(
    BaseEstimator, ClassifierMixin, MetaEstimatorMixin, metaclass=ABCMeta
):
    """Base class for Reweighting Biquality Classifiers.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self, final_estimator=None, random_state=None):
        self.final_estimator = final_estimator
        self.random_state = random_state

    def fit(self, X, y, sample_quality=None):
        """Fit the reweighted model.
        Parameters
        ----------
        X_trusted : array-like of shape (n_samples_trusted, n_features)
            The trusted samples.
        y_trusted : array-like of shape (n_samples_trusted,)
            The trusted targets.
        X_untrusted : array-like, shape (n_samples_untrusted, n_features)
            The unstruted samples.
        y_untrusted : array-like of shape (n_samples_untrusted,)
            The untrusted targets.
        Returns
        -------
        self : object
            Returns self.
        """

        X, y = self._validate_data(
            X, y, accept_sparse=["csr", "csc", "lil", "dok"], force_all_finite=False
        )

        check_classification_targets(y)

        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_classes_ = len(self.classes_)
        y = self._le.transform(y)

        random_state = check_random_state(self.random_state)

        self.sample_weight_ = np.ones(X.shape[0])

        self.sample_weight_[sample_quality == 0] = self._reweight(
            X[safe_mask(X, sample_quality == 1)],
            X[safe_mask(X, sample_quality == 0)],
        )

        if self.final_estimator is None:
            self.final_estimator_ = LogisticRegression()
        else:
            self.final_estimator_ = clone(self.final_estimator)

        if not has_fit_parameter(self.final_estimator_, "sample_weight"):
            raise ValueError(
                "%s doesn't support sample_weight."
                % self.final_estimator_.__class__.__name__
            )

        if hasattr(self.final_estimator_, "random_state"):
            self.final_estimator_.set_params(random_state=random_state)

        self.final_estimator_.fit(X, y, sample_weight=self.sample_weight_)

        # Return the classifier
        return self

    @abstractmethod
    def _reweight(self, X_trusted, X_untrusted, y_trusted, y_untrusted):
        """Implement reweighting scheme.
        Warning: This method needs to be overridden by subclasses.
        Parameters
        ----------
        X_trusted : array-like, shape (n_samples_trusted, n_features)
            The trusted samples.
        y_trusted : array-like, shape (n_samples_trusted,)
            The trusted targets.
        X_untrusted : array-like, shape (n_samples_untrusted, n_features)
            The unstruted samples.
        y_untrusted : array-like, shape (n_samples_untrusted,)
            The untrusted targets.
        Returns
        -------
        sample_weight_trusted : array-like of shape (n_samples,) or None
            The reweighted sample weights.
        sample_weight_untrusted : array-like of shape (n_samples,) or None
            The reweighted sample weights.
        """

    @if_delegate_has_method(delegate="final_estimator_")
    def decision_function(self, X):
        """Call decision function of the `final_estimator`.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        return self.final_estimator_.decision_function(X)

    @if_delegate_has_method(delegate="final_estimator_")
    def predict(self, X):
        """Predict the classes of `X`.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        return self._le.inverse_transform(self.final_estimator_.predict(X))

    @if_delegate_has_method(delegate="final_estimator_")
    def predict_proba(self, X):
        """Predict probability for each possible outcome.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.final_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate="final_estimator_")
    def predict_log_proba(self, X):
        """Predict log probability for each possible outcome.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        log_p : array, shape (n_samples, n_classes)
            Array with log prediction probabilities.
        """
        check_is_fitted(self)
        return self.final_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate="final_estimator")
    def score(self, X, y):
        """Call score on the `final_estimator`.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        y : array-like of shape (n_samples,)
            Array representing the labels.
        Returns
        -------
        score : float
            Result of calling score on the `final_estimator`.
        """
        check_is_fitted(self)
        return self.final_estimator_.score(X, y)


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
        final_estimator=None,
        *,
        kernel="rbf",
        kernel_params={},
        B=1000,
        epsilon=None,
        max_iter=1000,
        tol=1e-6,
        batch_size=0.05,
        n_jobs=None,
        random_state=None,
    ):
        super().__init__(final_estimator=final_estimator, random_state=random_state)

        self.kernel = kernel
        self.kernel_params = kernel_params
        self.B = B
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def _reweight(self, X_trusted, X_untrusted):
        n_samples_untrusted = _num_samples(X_untrusted)

        if isinstance(self.batch_size, float):
            if not 0 < self.batch_size <= 1:
                raise ValueError(
                    """When `batch_size` is provided as a float,
                    it needs to be between 0 (exclusive) and 1 (inclusive)."""
                )
            batch_size = int(self.batch_size * n_samples_untrusted)
        elif isinstance(self.batch_size, int):
            if not 1 <= self.batch_size:
                raise ValueError(
                    """When `batch_size` is provided as an int,
                    it needs to be superior or equal to 1."""
                )
            batch_size = self.batch_size
        elif self.batch_size is None:
            batch_size = n_samples_untrusted
        else:
            raise ValueError("""Unknown `batch_size` %s.""" % self.batch_size)

        if batch_size < 1 or batch_size > n_samples_untrusted:
            warnings.warn(
                """Computed `batch_size` is less than 1 or larger than
                the number of untrusted samples for this class.
                It is going to be clipped."""
            )
            batch_size = np.clip(batch_size, 1, n_samples_untrusted)

        batch_slices = gen_batches(n_samples_untrusted, batch_size)

        kmms = Parallel(n_jobs=self.n_jobs)(
            delayed(kmm)(
                X_untrusted[batch_slice],
                X_trusted,
                kernel=self.kernel,
                kernel_params=self.kernel_params,
                B=self.B,
                epsilon=self.epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                n_jobs=self.n_jobs,
            )
            for batch_slice in batch_slices
        )

        return np.concatenate(kmms).ravel()


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

    def __init__(
        self,
        base_estimator=None,
        final_estimator=None,
        method="probabilities",
        random_state=None,
    ):
        super().__init__(
            final_estimator=final_estimator,
            random_state=random_state,
        )

        self.base_estimator = base_estimator
        self.method = method

    def _reweight(self, X_trusted, X_untrusted):
        return pdr(X_untrusted, X_trusted, self.base_estimator, self.method).ravel()
