

def calculate_accuracy(y_preds,
                       y_true,
                       logits=True
                      ):
    """
    
    """
    return sum(torch.argmax(y_preds, dim=1) == y_true).item() / len(y_true)


def calculate_f1_score(y_preds,
                       y_true,
                       logits=True,
                       threshold=None
                      ):
    """
    
    """
    pass


def calculate_equal_error_rate(y_preds,
                               y_true,
                               logits=True
                              ):
    """
    
    """
    if logits:
        y_preds = (
            torch.nn.functional.softmax(y_preds, dim=1)[:, 1]
            if len(y_preds.size()) == 2
            else torch.nn.functional.sigmoid(y_preds)
        )
    elif len(y_preds.size()) == 2:
        y_preds = y_preds[:, 1]
    y_true = y_true.numpy()
    y_preds = y_preds.numpy()

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_preds)

    return sp.optimize.brentq(
        lambda x: 1.0 - x - sp.interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0
    )


if __name__ != "__main__":
    import numpy as np
    import scipy as sp
    import pandas as pd

    import torch
    import sklearn