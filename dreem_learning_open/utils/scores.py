import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score


def mask_non_scored_values(func):
    """ We do not considere unlabeled data to estimate metrics"""

    def lol(true, pred, *args, **kwargs):
        true = np.array(true)
        pred = np.array(pred)
        mask = (np.array([y in [0, 1, 2, 3, 4] for y in true]))
        return func(true[mask], pred[mask], *args, **kwargs)

    return lol


def f1_custom(true, pred):
    """
    We do a classical weighted f1 score:
    f1 score is computed for each class and then averages with number of occurence of each class
    BUT
    -> if a class has 0 example in TRUE we do not use it
    -> if a class is never predicted correctly, f1 = 0
    """
    f1s = []
    weights = []
    for i in range(5):
        N = (true == i).sum()
        if N == 0:  # If no example in True ignore the label
            continue
        TP = (true == pred)[true == i].sum()
        if TP == 0:
            f1 = 0
            f1s.append(f1)
            weights.append(N)
            continue
        FP = (true != pred)[pred == i].sum()
        FN = (true != pred)[true == i].sum()
        precision = TP / (TP + FP)  # cannot be nan because N > 0
        recall = TP / (TP + FN)  # cannot be nan because N > 0
        f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
        weights.append(N)
    return np.average(f1s, weights=weights)


def cohen_kappa_custom(true, pred):
    labels = list(set([0, 1, 2, 3, 4]).intersection(set(true)))
    return cohen_kappa_score(true, pred, labels=labels)


score_functions = {
    "accuracy": mask_non_scored_values(accuracy_score),
    "cohen_kappa": mask_non_scored_values(cohen_kappa_custom),
    "f1": mask_non_scored_values(f1_custom),
}
