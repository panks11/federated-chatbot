# utils/metrics.py

def accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true (List[int]): Ground truth labels
        y_pred (List[int]): Predicted labels

    Returns:
        float: Accuracy in range [0, 1]
    """
    if len(y_true) == 0:
        return 0.0

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)
