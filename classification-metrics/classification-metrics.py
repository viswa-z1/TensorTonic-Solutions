import numpy as np

def classification_metrics(y_true, y_pred,
                           average="micro",
                           pos_label=1):
    """
    Compute accuracy, precision, recall, and F1.

    Parameters:
        y_true     : true labels
        y_pred     : predicted labels
        average    : 'micro' | 'macro' | 'weighted' | 'binary'
        pos_label  : positive class for binary mode

    Returns:
        dict with:
        {
            "accuracy": float,
            "precision": float,
            "recall": float,
            "f1": float
        }
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    n = len(y_true)

    # Accuracy
    accuracy = float(np.mean(y_true == y_pred))

    # Unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)

    # Map labels to indices
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Confusion matrix
    cm = np.zeros((K, K), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        cm[class_to_idx[yt], class_to_idx[yp]] += 1

    # Per-class metrics
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []

    for i in range(K):

        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        support = np.sum(cm[i, :])

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)

    precision_list = np.array(precision_list)
    recall_list = np.array(recall_list)
    f1_list = np.array(f1_list)
    support_list = np.array(support_list)

    # ----- Averaging modes -----

    if average == "micro":

        TP = np.trace(cm)
        FP = np.sum(cm, axis=0) - np.diag(cm)
        FN = np.sum(cm, axis=1) - np.diag(cm)

        FP = np.sum(FP)
        FN = np.sum(FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

    elif average == "macro":

        precision = float(np.mean(precision_list))
        recall = float(np.mean(recall_list))
        f1 = float(np.mean(f1_list))

    elif average == "weighted":

        weights = support_list / np.sum(support_list)

        precision = float(np.sum(weights * precision_list))
        recall = float(np.sum(weights * recall_list))
        f1 = float(np.sum(weights * f1_list))

    elif average == "binary":

        if pos_label not in class_to_idx:
            raise ValueError("pos_label not found")

        idx = class_to_idx[pos_label]

        precision = float(precision_list[idx])
        recall = float(recall_list[idx])
        f1 = float(f1_list[idx])

    else:
        raise ValueError("Invalid average mode")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }