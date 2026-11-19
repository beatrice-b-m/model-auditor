"""Scalar metrics for fixed binary predictions.

Use metric instances with Auditor.set_metrics(). Direct data_call() calls expect
prepared columns: _truth (0/1), _pred (continuous score), _binary_pred (0/1), and
0/1 tp/tn/fp/fn confusion indicators as listed by each metric's inputs. Direct
calls do not perform the Auditor's full validation and never modify the frame.
Rates are fractions, not percentages. Undefined ratios return NaN. Counts have
ci_eligible=False; other built-ins request pointwise intervals via the Auditor.

Example:
    >>> import pandas as pd
    >>> from model_auditor.metrics import Sensitivity
    >>> Sensitivity().data_call(pd.DataFrame({"tp": [1, 0], "fn": [0, 1]}))
    0.5"""

from typing import Literal as _Literal
from typing import Protocol, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray as _NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return a stable ratio, returning NaN when the denominator is zero."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


class AuditorMetric(Protocol):
    """Structural interface for scalar metrics; subclassing is optional.

    Attributes:
        name: Unique stable metric ID; duplicates are rejected by set_metrics().
        label: Display label for tables and plots.
        inputs: Required prepared column names: _truth, _pred, _binary_pred,
            discovered confusion inputs, or existing input data columns.
        ci_eligible: Whether the Auditor should attempt an interval.

    Notes:
        Optional implementation metadata includes direction ("higher", "lower",
        "none"), parameters (configuration dict), binomial_columns (success and
        failure indicator names), or binomial_indicator (a 0/1 column). The last two
        select analytic IID rate intervals; supply them only for binomial rates.
        An optional scale (finite positive number, default 1.0) multiplies the
        reported estimate, interval bounds and difference contrasts, for example
        scale=1000 for a rate per 1,000 rows; denominators, support counts and
        ratio contrasts are unchanged, and a non-unit scale is recorded in result
        parameters and metadata. data_call itself still returns the unscaled value.
        These are optional capabilities, not required protocol members.
        data_call must return a scalar without mutating data, because every
        metric on a level reads the same resampled rows; undefined values should
        be NaN. The Auditor skips calls for empty subgroups.

    Example:
        >>> class Accuracy:
        ...     name, label = "accuracy", "Accuracy"
        ...     inputs, ci_eligible = ["tp", "tn"], True
        ...     direction = "higher"
        ...     def data_call(self, data: pd.DataFrame) -> float:
        ...         return float((data["tp"].sum() + data["tn"].sum()) / len(data))
        >>> Accuracy().name
        'accuracy'"""

    name: str
    label: str
    inputs: list[str]
    ci_eligible: bool

    def data_call(self, data: pd.DataFrame) -> Union[float, int, np.integer]:
        """Compute one scalar from prepared evaluation rows without modifying them.

        Args:
            data: DataFrame containing the columns listed in inputs. _truth and
                _binary_pred are binary; confusion columns are 0/1 indicators.

        Returns:
            Numeric estimate (counts may be NumPy integer scalars), or NaN
            when mathematically undefined.

        Raises:
            NotImplementedError: This protocol method has no implementation.
            KeyError: Implementations generally raise for absent required columns."""
        raise NotImplementedError


class Sensitivity(AuditorMetric):
    """Sensitivity (True Positive Rate) metric.

    Calculates TP / (TP + FN), the proportion of actual positives
    correctly identified.
    """

    binomial_columns: tuple[str, str] = ("tp", "fn")
    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "sensitivity"
    label: str = "Sensitivity"
    inputs: list[str] = ["tp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate sensitivity from the data.

        Args:
            data: DataFrame with 'tp' and 'fn' columns.

        Returns:
            Sensitivity value between 0 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        n_tp: int = data["tp"].sum()
        n_fn: int = data["fn"].sum()
        return _safe_ratio(n_tp, n_tp + n_fn)


class Specificity(AuditorMetric):
    """Specificity (True Negative Rate) metric.

    Calculates TN / (TN + FP), the proportion of actual negatives
    correctly identified.
    """

    binomial_columns: tuple[str, str] = ("tn", "fp")
    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "specificity"
    label: str = "Specificity"
    inputs: list[str] = ["tn", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate specificity from the data.

        Args:
            data: DataFrame with 'tn' and 'fp' columns.

        Returns:
            Specificity value between 0 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        n_tn: int = data["tn"].sum()
        n_fp: int = data["fp"].sum()
        return _safe_ratio(n_tn, n_tn + n_fp)


class Precision(AuditorMetric):
    """Precision (Positive Predictive Value) metric.

    Calculates TP / (TP + FP), the proportion of positive predictions
    that are correct.
    """

    binomial_columns: tuple[str, str] = ("tp", "fp")
    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "precision"
    label: str = "Precision"
    inputs: list[str] = ["tp", "fp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate precision from the data.

        Args:
            data: DataFrame with 'tp' and 'fp' columns.

        Returns:
            Precision value between 0 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        n_tp: int = data["tp"].sum()
        n_fp: int = data["fp"].sum()
        return _safe_ratio(n_tp, n_tp + n_fp)


class Recall(Sensitivity):
    """Recall metric (alias for Sensitivity).

    Calculates TP / (TP + FN), identical to Sensitivity.
    """

    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "recall"
    label: str = "Recall"


class F1Score(AuditorMetric):
    """F1 Score metric.

    Calculates the harmonic mean of precision and recall:
    2 * (precision * recall) / (precision + recall).
    """

    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "f1"
    label: str = "F1 Score"
    inputs: list[str] = ["tp", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate F1 score from the data.

        Args:
            data: DataFrame with 'tp', 'fp', and 'fn' columns.

        Returns:
            F1 score value between 0 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        tp = float(data["tp"].sum())
        fp = float(data["fp"].sum())
        fn = float(data["fn"].sum())
        return _safe_ratio(2 * tp, 2 * tp + fp + fn)


class AUROC(AuditorMetric):
    """Area Under the Receiver Operating Characteristic curve metric.

    Uses sklearn's roc_auc_score to compute AUROC from continuous
    predictions and binary ground truth.
    """

    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "auroc"
    label: str = "AUROC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate AUROC from the data.

        Args:
            data: DataFrame with '_truth' and '_pred' columns.

        Returns:
            AUROC value between 0 and 1, or NaN if calculation fails.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.
            Without both truth classes the result is NaN.

        Raises:
            KeyError: A required input column is absent."""
        if data["_truth"].nunique() < 2:
            return float("nan")
        try:
            return float(roc_auc_score(data["_truth"], data["_pred"]))
        except ValueError:
            return float("nan")


class AUPRC(AuditorMetric):
    """Compatibility name for non-interpolated average precision (not trapezoidal area).

    Uses sklearn's average_precision_score to compute AUPRC from
    continuous predictions and binary ground truth.
    """

    direction: _Literal["higher", "lower", "none"] = "higher"
    parameters: dict[str, str] = {"integration": "average_precision"}
    name: str = "auprc"
    label: str = "AUPRC"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate AUPRC from the data.

        Args:
            data: DataFrame with '_truth' and '_pred' columns.

        Returns:
            AUPRC value between 0 and 1, or NaN if calculation fails.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.
            With no positive truth the result is NaN. This is average precision,
            not trapezoidal area under the precision-recall curve.

        Raises:
            KeyError: A required input column is absent."""
        if data["_truth"].sum() == 0:
            return float("nan")
        try:
            return float(average_precision_score(data["_truth"], data["_pred"]))
        except ValueError:
            return float("nan")


class MatthewsCorrelationCoefficient(AuditorMetric):
    """Matthews Correlation Coefficient (MCC) metric.

    A balanced measure that accounts for all four confusion matrix values.
    Returns values between -1 (total disagreement) and +1 (perfect prediction),
    with 0 indicating zero measured correlation, not proof of randomness.
    """

    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "mcc"
    label: str = "Matthews Correlation Coefficient"
    inputs: list[str] = ["tp", "tn", "fp", "fn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate MCC from the data.

        Args:
            data: DataFrame with 'tp', 'tn', 'fp', and 'fn' columns.

        Returns:
            MCC value between -1 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        tp = float(data["tp"].sum())
        tn = float(data["tn"].sum())
        fp = float(data["fp"].sum())
        fn = float(data["fn"].sum())

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denominator == 0:
            return float("nan")
        return float(numerator / denominator)


class FBetaScore(AuditorMetric):
    """F-beta Score metric with configurable beta parameter.

    Generalizes F1 score by allowing different weightings of precision
    and recall. Beta < 1 weights precision higher, beta > 1 weights
    recall higher.
    """

    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "fbeta"
    label: str = "F-beta Score"
    inputs: list[str] = ["tp", "fp", "fn"]
    ci_eligible: bool = True

    def __init__(self, beta: float = 1.0) -> None:
        """Configure recall weighting and a beta-specific name, e.g. "f1_0".

        Args:
            beta: Finite positive weight; default 1.0 (F1). Values below 1 emphasize
                precision and above 1 emphasize recall.

        Raises:
            ValueError: beta is nonfinite or not positive."""
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive.")
        self.beta: float = float(beta)
        self.parameters: dict[str, float] = {"beta": self.beta}
        self.name = f"f{self.beta}".replace(".", "_")
        self.label = f"F{self.beta} Score"

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate F-beta score from the data.

        Args:
            data: DataFrame with columns needed by Precision and Recall.

        Returns:
            F-beta score value between 0 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        tp = float(data["tp"].sum())
        fp = float(data["fp"].sum())
        fn = float(data["fn"].sum())
        beta_sq = self.beta**2
        return _safe_ratio((1 + beta_sq) * tp, (1 + beta_sq) * tp + beta_sq * fn + fp)


class TPR(Sensitivity):
    """True Positive Rate metric (alias for Sensitivity)."""

    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "tpr"
    label: str = "TPR"


class TNR(Specificity):
    """True Negative Rate metric (alias for Specificity)."""

    direction: _Literal["higher", "lower", "none"] = "higher"
    name: str = "tnr"
    label: str = "TNR"


class FPR(AuditorMetric):
    """False Positive Rate metric.

    Calculates FP / (FP + TN), the proportion of actual negatives
    incorrectly identified as positive.
    """

    binomial_columns: tuple[str, str] = ("fp", "tn")
    direction: _Literal["higher", "lower", "none"] = "lower"
    name: str = "fpr"
    label: str = "FPR"
    inputs: list[str] = ["fp", "tn"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate FPR from the data.

        Args:
            data: DataFrame with 'fp' and 'tn' columns.

        Returns:
            FPR value between 0 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        n_fp: int = data["fp"].sum()
        n_tn: int = data["tn"].sum()
        return _safe_ratio(n_fp, n_fp + n_tn)


class FNR(AuditorMetric):
    """False Negative Rate metric.

    Calculates FN / (FN + TP), the proportion of actual positives
    incorrectly identified as negative.
    """

    binomial_columns: tuple[str, str] = ("fn", "tp")
    direction: _Literal["higher", "lower", "none"] = "lower"
    name: str = "fnr"
    label: str = "FNR"
    inputs: list[str] = ["fn", "tp"]
    ci_eligible: bool = True

    def data_call(self, data: pd.DataFrame) -> float:
        """Calculate FNR from the data.

        Args:
            data: DataFrame with 'fn' and 'tp' columns.

        Returns:
            FNR value between 0 and 1.

        Notes:
            Returns NaN when the formula's denominator is zero. Does not mutate data.
            Direct calls require nonmissing binary confusion indicators; evaluation
            through Auditor validates truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        n_fn: int = data["fn"].sum()
        n_tp: int = data["tp"].sum()
        return _safe_ratio(n_fn, n_fn + n_tp)


class nData(AuditorMetric):
    """Sample size metric.

    Returns the number of rows in the data subset.
    """

    direction: _Literal["higher", "lower", "none"] = "none"
    name: str = "n"
    label: str = "N"
    inputs: list[str] = []
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int:
        """Return the number of samples in the data.

        Args:
            data: DataFrame to count rows from.

        Returns:
            Number of rows in the DataFrame.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first."""
        return len(data)


class nTP(AuditorMetric):
    """True Positive count metric.

    Returns the total number of true positives in the data subset.
    """

    direction: _Literal["higher", "lower", "none"] = "none"
    name: str = "n_tp"
    label: str = "TP"
    inputs: list[str] = ["tp"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int | np.integer:
        """Return the count of true positives.

        Args:
            data: DataFrame with 'tp' column.

        Returns:
            Sum of true positives.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        return data["tp"].sum()


class nTN(AuditorMetric):
    """True Negative count metric.

    Returns the total number of true negatives in the data subset.
    """

    direction: _Literal["higher", "lower", "none"] = "none"
    name: str = "n_tn"
    label: str = "TN"
    inputs: list[str] = ["tn"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int | np.integer:
        """Return the count of true negatives.

        Args:
            data: DataFrame with 'tn' column.

        Returns:
            Sum of true negatives.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        return data["tn"].sum()


class nFP(AuditorMetric):
    """False Positive count metric.

    Returns the total number of false positives in the data subset.
    """

    direction: _Literal["higher", "lower", "none"] = "none"
    name: str = "n_fp"
    label: str = "FP"
    inputs: list[str] = ["fp"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int | np.integer:
        """Return the count of false positives.

        Args:
            data: DataFrame with 'fp' column.

        Returns:
            Sum of false positives.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        return data["fp"].sum()


class nFN(AuditorMetric):
    """False Negative count metric.

    Returns the total number of false negatives in the data subset.
    """

    direction: _Literal["higher", "lower", "none"] = "none"
    name: str = "n_fn"
    label: str = "FN"
    inputs: list[str] = ["fn"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int | np.integer:
        """Return the count of false negatives.

        Args:
            data: DataFrame with 'fn' column.

        Returns:
            Sum of false negatives.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        return data["fn"].sum()


class nPositive(AuditorMetric):
    """Positive class count metric.

    Returns the number of actual positive cases in the ground truth.
    """

    direction: _Literal["higher", "lower", "none"] = "none"
    name: str = "n_pos"
    label: str = "Pos."
    inputs: list[str] = ["_truth"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int | np.integer:
        """Return the count of positive ground truth labels.

        Args:
            data: DataFrame with '_truth' column.

        Returns:
            Number of rows where _truth equals 1.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        return (data["_truth"] == 1).astype(int).sum()


class nNegative(AuditorMetric):
    """Negative class count metric.

    Returns the number of actual negative cases in the ground truth.
    """

    direction: _Literal["higher", "lower", "none"] = "none"
    name: str = "n_neg"
    label: str = "Neg."
    inputs: list[str] = ["_truth"]
    ci_eligible: bool = False

    def data_call(self, data: pd.DataFrame) -> int | np.integer:
        """Return the count of negative ground truth labels.

        Args:
            data: DataFrame with '_truth' column.

        Returns:
            Number of rows where _truth equals 0.

        Notes:
            Does not modify data. Direct calls expect nonmissing prepared inputs;
            use Auditor evaluation to validate truth and scores first.

        Raises:
            KeyError: A required input column is absent."""
        return (data["_truth"] == 0).astype(int).sum()


class AveragePrecision(AUPRC):
    """Non-interpolated average precision; AUPRC remains a compatibility alias.

    Prevalence-dependent. Undefined without positives. This is not trapezoidal
    PR area and is not a probability calibration metric.
    """

    name: str = "average_precision"
    label: str = "Average precision"


class NegativePredictiveValue(AuditorMetric):
    """TN / (TN + FN); undefined without negative predictions."""

    name: str = "npv"
    label: str = "NPV"
    inputs: list[str] = ["tn", "fn"]
    ci_eligible: bool = True
    direction: _Literal["higher", "lower", "none"] = "higher"
    binomial_columns: tuple[str, str] = ("tn", "fn")

    def data_call(self, data: pd.DataFrame) -> float:
        """Compute negative predictive value, TN/(TN+FN).

        Args:
            data: Prepared DataFrame with 'tn' and 'fn' indicator columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Fraction in [0, 1], or NaN with no negative predictions.

        Raises:
            KeyError: A required input column is absent."""
        return _safe_ratio(data["tn"].sum(), data["tn"].sum() + data["fn"].sum())


class BalancedAccuracy(AuditorMetric):
    """Mean sensitivity and specificity; requires both truth classes."""

    name: str = "balanced_accuracy"
    label: str = "Balanced accuracy"
    inputs: list[str] = ["tp", "tn", "fp", "fn"]
    ci_eligible: bool = True
    direction: _Literal["higher", "lower", "none"] = "higher"

    def data_call(self, data: pd.DataFrame) -> float:
        """Compute the mean of sensitivity and specificity.

        Args:
            data: Prepared DataFrame with 'tp', 'tn', 'fp' and 'fn' indicator columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Fraction in [0, 1], or NaN if either truth class is absent.

        Raises:
            KeyError: A required input column is absent."""
        return (Sensitivity().data_call(data) + Specificity().data_call(data)) / 2


class Prevalence(AuditorMetric):
    """Observed positive-class proportion; descriptive, not a performance rank."""

    name: str = "prevalence"
    label: str = "Prevalence"
    inputs: list[str] = ["_truth"]
    ci_eligible: bool = True
    direction: _Literal["higher", "lower", "none"] = "none"
    binomial_indicator: str = "_truth"

    def data_call(self, data: pd.DataFrame) -> float:
        """Compute the observed positive-class fraction.

        Args:
            data: Prepared DataFrame with binary '_truth' columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Fraction in [0, 1], or NaN for empty data.

        Raises:
            KeyError: A required input column is absent."""
        return float(data["_truth"].mean())


class SelectionRate(Prevalence):
    """Predicted-positive proportion at the chosen policy threshold."""

    name: str = "selection_rate"
    label: str = "Selection rate"
    inputs: list[str] = ["_binary_pred"]
    binomial_indicator: str = "_binary_pred"

    def data_call(self, data: pd.DataFrame) -> float:
        """Compute the predicted-positive fraction.

        Args:
            data: Prepared DataFrame with binary '_binary_pred' columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Fraction in [0, 1], or NaN for empty data.

        Raises:
            KeyError: A required input column is absent."""
        return float(data["_binary_pred"].mean())


def validate_probabilities(data: pd.DataFrame) -> _NDArray[np.float64]:
    """Extract a float64 prediction array and validate probability bounds.

    Args:
        data: DataFrame with a numeric-convertible _pred column.

    Returns:
        One-dimensional float64 array of finite values in [0, 1]. May share storage
        with data; do not mutate it unless that is intended. Empty arrays are valid.

    Raises:
        KeyError: _pred is absent.
        ValueError: Values cannot be converted, are nonfinite or outside [0, 1]."""
    probabilities = data["_pred"].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Probability metrics require finite scores in [0, 1].")
    return probabilities


class BrierScore(AuditorMetric):
    """Mean squared probability error; measures calibration and discrimination."""

    name: str = "brier_score"
    label: str = "Brier score"
    inputs: list[str] = ["_truth", "_pred"]
    ci_eligible: bool = True
    direction: _Literal["higher", "lower", "none"] = "lower"

    def data_call(self, data: pd.DataFrame) -> float:
        """Compute mean squared probability error.

        Args:
            data: Prepared DataFrame with '_truth' (0/1) and '_pred' (probability) columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Mean (prediction - truth)**2 in [0, 1]; lower is better.
            NaN for empty data.

        Raises:
            KeyError: A required input column is absent.
            ValueError: Predictions are nonfinite or outside [0, 1]."""
        probabilities = validate_probabilities(data)
        return float(np.mean((probabilities - data["_truth"].to_numpy()) ** 2))


class LogLoss(BrierScore):
    """Binary log loss, clipping probabilities to float64 epsilon at endpoints."""

    name: str = "log_loss"
    label: str = "Log loss"
    parameters: dict[str, float] = {"clip_epsilon": float(np.finfo(float).eps)}

    def data_call(self, data: pd.DataFrame) -> float:
        """Compute binary cross-entropy using natural logarithms.

        Args:
            data: Prepared DataFrame with '_truth' (0/1) and '_pred' (probability) columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Nonnegative mean loss in nats; lower is better. Probabilities are
            clipped to float64 epsilon and 1-epsilon after validation. Empty
            data returns NaN.

        Raises:
            KeyError: A required input column is absent.
            ValueError: Predictions are nonfinite or outside [0, 1]."""
        p = np.clip(
            validate_probabilities(data), np.finfo(float).eps, 1 - np.finfo(float).eps
        )
        y = data["_truth"].to_numpy()
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def _calibration_fit(data: pd.DataFrame, fit_slope: bool) -> float:
    """Logistic calibration with explicit rejection of unidentified/separated fits."""
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    p = validate_probabilities(data)
    y = data["_truth"].to_numpy(dtype=float)
    if len(np.unique(y)) != 2 or np.any((p == 0) | (p == 1)):
        return float("nan")
    x = logit(p)
    if fit_slope:
        if (
            np.ptp(x) == 0
            or x[y == 0].max() <= x[y == 1].min()
            or x[y == 1].max() <= x[y == 0].min()
        ):
            return float("nan")
        design = np.column_stack([np.ones(len(x)), x])
        offset = np.zeros(len(x))
        initial = np.array([0.0, 1.0])
    else:
        design = np.ones((len(x), 1))
        offset = x
        initial = np.array([0.0])

    def objective(beta):
        eta = design @ beta + offset
        return np.mean(np.logaddexp(0, eta) - y * eta), design.T @ (
            expit(eta) - y
        ) / len(y)

    fit = minimize(objective, initial, jac=True, method="BFGS", options={"gtol": 1e-8})
    if not fit.success or not np.isfinite(fit.x).all():
        return float("nan")
    return float(fit.x[-1])


class CalibrationIntercept(BrierScore):
    """Calibration-in-the-large: logistic intercept with logit(score) as offset.

    Ideal value is zero. Requires interior probabilities and both truth classes.
    Not a higher/lower-is-better metric. Invalid fits return NaN.
    """

    name: str = "calibration_intercept"
    label: str = "Calibration intercept"
    direction: _Literal["higher", "lower", "none"] = "none"

    def data_call(self, data: pd.DataFrame) -> float:
        """Fit calibration-in-the-large with logit(score) as a fixed offset.

        Args:
            data: Prepared DataFrame with '_truth' (0/1) and '_pred' (probability) columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Log-odds intercept (ideal 0). NaN for absent classes, endpoint
            probabilities, unsuccessful optimization or nonfinite coefficients.

        Raises:
            KeyError: A required input column is absent.
            ValueError: Predictions are nonfinite or outside [0, 1]."""
        return _calibration_fit(data, fit_slope=False)


class CalibrationSlope(CalibrationIntercept):
    """Slope from logistic regression on logit(score), with a fitted intercept.

    Ideal value is one. Constant scores, separation and boundary probabilities
    are not estimable; bootstrap failures are exposed in interval diagnostics.
    """

    name: str = "calibration_slope"
    label: str = "Calibration slope"

    def data_call(self, data: pd.DataFrame) -> float:
        """Fit logistic truth on logit(score) with an estimated intercept.

        Args:
            data: Prepared DataFrame with '_truth' (0/1) and '_pred' (probability) columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Dimensionless slope (ideal 1). NaN for absent classes, endpoint or
            constant probabilities, separated data, failed optimization or
            nonfinite coefficients. The fitted intercept is not returned.

        Raises:
            KeyError: A required input column is absent.
            ValueError: Predictions are nonfinite or outside [0, 1]."""
        return _calibration_fit(data, fit_slope=True)


class ExpectedCost(AuditorMetric):
    """Mean false-positive/false-negative cost per evaluated row.

    Args:
        false_positive_cost: Finite nonnegative cost for each false positive;
            default 1. Use the same units as false_negative_cost.
        false_negative_cost: Finite nonnegative cost for each false negative;
            default 1. Correct predictions incur zero cost.

    Attributes:
        parameters: Mutable dict holding the two float costs.

    Raises:
        ValueError: Either cost is negative or nonfinite.

    Notes:
        Lower is better. Undefined for empty data. Multiple configurations share
        name="expected_cost"; assign distinct names before configuring both."""

    name: str = "expected_cost"
    label: str = "Expected cost"
    inputs: list[str] = ["fp", "fn"]
    ci_eligible: bool = True
    direction: _Literal["higher", "lower", "none"] = "lower"

    def __init__(
        self, false_positive_cost: float = 1, false_negative_cost: float = 1
    ) -> None:
        """Store costs for incorrect decisions; correct decisions have zero cost.

        Args:
            false_positive_cost: Finite nonnegative cost; default 1.
            false_negative_cost: Finite nonnegative cost in the same units; default 1.

        Raises:
            ValueError: A cost is negative or nonfinite."""
        if not all(
            np.isfinite(v) and v >= 0
            for v in (false_positive_cost, false_negative_cost)
        ):
            raise ValueError("Costs must be finite and nonnegative.")
        self.parameters: dict[str, float] = {
            "false_positive_cost": float(false_positive_cost),
            "false_negative_cost": float(false_negative_cost),
        }

    def data_call(self, data: pd.DataFrame) -> float:
        """Compute (FP * false_positive_cost + FN * false_negative_cost) / N.

        Args:
            data: Prepared DataFrame with 'fp' and 'fn' indicator columns.
                Not modified. Missing input values are unsupported; use Auditor
                evaluation for input validation.

        Returns:
            Mean cost per row in the supplied cost units, or NaN for empty data.

        Raises:
            KeyError: A required input column is absent."""
        return _safe_ratio(
            self.parameters["false_positive_cost"] * data["fp"].sum()
            + self.parameters["false_negative_cost"] * data["fn"].sum(),
            len(data),
        )
