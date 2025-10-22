import numpy as np

def accuracy(y_true, y_pred, atol=1e-6, rtol=0.0):
    """
    Returns the fraction of examples where prediction numerically equals the label,
    using np.isclose (handles floats from division). Works for ints or floats.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    # guard against NaNs/Infs in predictions
    finite = np.isfinite(yt) & np.isfinite(yp)
    if not finite.all():
        # treat non-finite preds as mismatches
        mask = finite
        correct = np.zeros_like(yt, dtype=bool)
        correct[mask] = np.isclose(yt[mask], yp[mask], atol=atol, rtol=rtol)
    else:
        correct = np.isclose(yt, yp, atol=atol, rtol=rtol)

    return float(correct.mean())
