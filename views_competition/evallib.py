"""Evaluation metrics library."""

# TODO: clean up docstrings, lint.

from typing import Optional, Union, Callable
import numpy as np


def pos_neg_bin(val: float) -> int:
    """Helper for tadda_score:
    Check if val is to the right or left of 0
    Args:
      val -- a float to check
    Returns:
      -1 if val<0
       1 if val>0
       np.NaN if val==0
    """
    if val < 0:
        return -1
    elif val > 0:
        return 1
    else:
        return np.NaN


def pos_z_neg(
    val: float, epsilon: float, middle_closed_interval: bool = True
) -> int:
    """Helper for tadda_score:
    Check if val to right (0), inside (1), or to the left (2)
    of pm epsilon interval
     Note that 0 maps to -1 in the textual description, 1 to 0 and 2 to 1.
    Args:
      val -- a value to check
      epsilon -- defines the bounds (-epsilon, +epsilon)
      middle_closed_interval -- by default middle interval is closed (includes endpoints -epsilon and +epsilon)
    Returns
      An int: by default, 0 if val>-epsilon; 1 if -epsilon<=val<=epsilon, 2 if val<epsilon
    """
    if middle_closed_interval:
        if val < -epsilon:
            return 0
        elif -epsilon >= val or val <= epsilon:
            return 1
        elif val > epsilon:
            return 2
        else:
            raise ValueError("val and epsilon combination is not as expected")
    elif not middle_closed_interval:
        if val <= -epsilon:
            return 0
        elif -epsilon > val or val < epsilon:
            return 1
        elif val >= epsilon:
            return 2
        else:
            raise ValueError("val and epsilon combination is not as expected")
    else:
        raise ValueError(
            "unexpected value for middle_closed_interval passed into function"
        )


def make_lookup_table(epsilon):
    """Helper for tadda_score:
    Make table to fill in correct x in f - x, where x is defined in this table
    for use calc_tadda_point and fed into wrong_dir_penalty, None means no dir penalty is applied.
    """
    # dims are d, y, f and rows/col order are 0: lower than -e, 1: mid, 2:higher than +e
    if epsilon is None:
        return None
    else:
        lookup_tab = np.array(
            [
                [
                    [np.NaN, np.NaN, epsilon],
                    [np.NaN, np.NaN, np.NaN],
                    [-epsilon, np.NaN, np.NaN],
                ],
                [
                    [np.NaN, -epsilon, -epsilon],
                    [-epsilon, np.NaN, epsilon],
                    [epsilon, epsilon, np.NaN],
                ],
            ]
        )
        return lookup_tab


def wrong_dir_penalty(
    y: float,
    f: float,
    dist: Callable,
    d: int,
    epsilon: float,
    lookup_table: np.ndarray,
    middle_closed_interval: bool = True,
) -> float:
    """Helper for tadda_score: lookup correct penalty."""
    if d > 0:
        loc = (
            d - 1,
            pos_z_neg(
                val=y,
                epsilon=epsilon,
                middle_closed_interval=middle_closed_interval,
            ),
            pos_z_neg(
                val=f,
                epsilon=epsilon,
                middle_closed_interval=middle_closed_interval,
            ),
        )
        val = lookup_table[loc]
        return 0 if np.isnan(val) else dist(f - val)
    elif d == 0:
        y_bsgn = pos_neg_bin(y)
        f_bsgn = pos_neg_bin(f)
        if np.isnan(y_bsgn) or np.isnan(f_bsgn):
            raise ValueError(
                "for d=0, neither y nor f can hold 0 values, try d=1 or d=2"
            )
        else:
            return 0 if y_bsgn == f_bsgn else dist(f)
    else:
        raise ValueError(f"d must be 0, 1, or 2; recieved d={d}")


def tadda_GFFO(
    y_true: np.array,
    y_pred: np.array,
    d: Optional[int] = 2,
    epsilon: Optional[float] = 0.048,
    absolute_dist: bool = True,
    middle_closed_interval: bool = True,
    element_by_element: bool = False,
    loud: bool = False,
) -> Union[float, np.array]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(y_true, y_pred)

    wrong_signs1 = (np.sign(y_pred) * np.sign(y_true) == -1).astype("int")  # differnt sing
    wrong_signs2 = ((np.sign(y_pred) == 0).astype("int")) * (
        (np.abs(np.sign(y_true)) > 0).astype("int"))  # pred == 0 and true <> 0
    wrong_signs3 = ((np.sign(y_true) == 0).astype("int")) * (
        (np.abs(np.sign(y_pred)) > 0).astype("int"))  # true == 0 and pred <> 0
    wrong_signs = ((wrong_signs1 + wrong_signs2 + wrong_signs3) > 0).astype("int")

    significant_diff = (np.abs(y_true - y_pred) > epsilon).astype("int")
    sign_loss = (2 * mae * wrong_signs * significant_diff).mean()

    return mae + sign_loss


def tadda_score(
    y_deltas: np.array,
    f_deltas: np.array,
    d: Optional[int] = 2,
    epsilon: Optional[float] = 0.048,
    absolute_dist: bool = True,
    middle_closed_interval: bool = True,
    element_by_element: bool = False,
    loud: bool = False,
) -> Union[float, np.array]:
    """Calculate TADDA for point predictions
    inputs:
        y -- 1d np.ndarray of length N holding the actual changes
        f -- 1d np.ndarray of length N holding the forecasted values
        d -- which directional penalty to use
            0 -- if epsilon is None, ignores y=0 cases, if pr(y=0)=0 this is
                reasonable (Default), when epsilon is not None |y|<epsilon
                cases ignored, pr(|y|=epsilon)=0 should hold
            1 -- a non-zero epsilon is required, pr(y approx 0)>0 is possible,
                treats near zero vals (|y|<epsilon) as both pos and neg
            2 -- a non-zero epsilon is required, pr(y approx 0)>0 is possible,
                treats near zero vals (|y|<epsilon) as neither pos nor neg
        epsilon  -- a positive scalar that defines the interval around zero
            where values are "near zero", epsilon is ignored if d=0 (default is
            None, must by non-None when d is 1 or 2)
        absolute_dist -- should L1 (True by default) or squared L2 be used
            (False)
        element_by_element -- return the mean of the individual contributions
            if False
            or the vector of individual TADDA values if True (False by default)
        loud -- print out term1 and term2 contributions (True), do not (False
            by default)
    outputs:
        single float if element_by_element is False, and np.array if
        element_by_element is True
    """
    if epsilon is not None:
        assert epsilon > 0.0, "epsilon must be greater than 0.0"

    def dist_1(x):
        return np.abs(x)

    def dist_2(x):
        return np.power(x, 2)

    if absolute_dist:
        dist_fun = dist_1
    else:
        dist_fun = dist_2
    term1 = dist_fun(y_deltas - f_deltas)
    term2 = np.empty_like(y_deltas, dtype=np.float64)
    # just make lookup_table once per call, not for each fi
    lookup_table = make_lookup_table(epsilon)
    for yi, fi, i in zip(y_deltas, f_deltas, range(len(y_deltas))):
        term2[i] = wrong_dir_penalty(
            y=yi,
            f=fi,
            d=d,
            epsilon=epsilon,
            lookup_table=lookup_table,
            dist=dist_fun,
            middle_closed_interval=middle_closed_interval,
        )
    if loud:
        print(term1, term2)
    if element_by_element:
        return term1 + term2
    return (term1 + term2).mean()


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient.
    The concordance correlation coefficient is a measure of inter-rater agreement.
    It measures the deviation of the relationship between predicted and true values
    from the 45 degree angle.
    Read more: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Original paper: Lawrence, I., and Kuei Lin. "A concordance correlation coefficient to evaluate reproducibility." Biometrics (1989): 255-268.
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : A float in the range [-1,1]. A value of 1 indicates perfect agreement
    between the true and the predicted values.
    Examples
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> concordance_correlation_coefficient(y_true, y_pred)
    0.97678916827853024
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator
