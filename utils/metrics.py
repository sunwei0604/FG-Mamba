import numpy as np
from sklearn.metrics import confusion_matrix


def metrics(prediction, target, n_classes=None):
    prediction = np.array(prediction)
    target = np.array(target)
    ignored_mask = np.zeros(target.shape, dtype=bool)
    ignored_mask[target < 0] = True
    valid_mask = ~ignored_mask
    target = target[valid_mask]
    prediction = prediction[valid_mask]
    results = {}
    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    cm = confusion_matrix(target, prediction, labels=range(n_classes))
    results["Confusion matrix"] = cm
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)
    results["Accuracy"] = accuracy * 100.0
    class_acc = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            acc = cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            acc = 0.
        class_acc[i] = acc
    results["class acc"] = class_acc * 100.0
    results['AA'] = np.mean(class_acc) * 100.0
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100.0
    return results
