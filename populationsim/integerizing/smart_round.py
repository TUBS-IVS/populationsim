import numpy as np


def smart_round(int_weights, resid_weights, target_sum, tie_keys=None):
    """
    Round weights while ensuring (as far as possible that result sums to target_sum)

    Parameters
    ----------
    int_weights : numpy.ndarray(int)
    resid_weights : numpy.ndarray(float)
    target_sum : int
    tie_keys : numpy.ndarray(float) or None
        fork option (INTEGERIZER_DRAW): secondary sort key. None (default) keeps
        the upstream order np.argsort(resid_weights). When given, rows are
        ordered by np.lexsort((tie_keys, resid_weights)): by resid_weights
        first and, among equal resid_weights (e.g. the 0/1 values of an LP
        solution), by tie_keys, so the rows rounded up follow the draw keys
        instead of the row position.

    Returns
    -------
    rounded_weights : numpy.ndarray array of ints
    """
    assert len(int_weights) == len(resid_weights)
    if tie_keys is not None and len(tie_keys) != len(resid_weights):
        raise ValueError(
            "smart_round: %s tie_keys for %s resid_weights"
            % (len(tie_keys), len(resid_weights))
        )
    assert (int_weights == int_weights.astype(int)).all()
    assert target_sum == int(target_sum)

    target_sum = int(target_sum)

    # integer part of numbers to round (astype both copies and coerces)
    rounded_weights = int_weights.astype(int)

    # find number of residuals that we need to round up
    int_shortfall = target_sum - rounded_weights.sum()

    # clip to feasible, in case target was not achievable by rounding
    int_shortfall = np.clip(int_shortfall, 0, len(resid_weights))

    # Order the residual weights and round at the tipping point where target_sum is achieved
    if int_shortfall > 0:
        if tie_keys is None:
            # indices of the int_shortfall highest resid_weights
            i = np.argsort(resid_weights)[-int_shortfall:]
        else:
            # fork: highest resid_weights, ties broken by the highest tie_keys
            i = np.lexsort((tie_keys, resid_weights))[-int_shortfall:]

        # add 1 to the integer weights that we want to round upwards
        rounded_weights[i] += 1

    return rounded_weights
