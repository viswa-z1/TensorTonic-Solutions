def compute_monitoring_metrics(system_type, y_true, y_pred):
    metrics = []

    if system_type == "classification":
        # Calculate TP, FP, FN, TN
        TP = sum((true == 1 and pred == 1) for true, pred in zip(y_true, y_pred))
        FP = sum((true == 0 and pred == 1) for true, pred in zip(y_true, y_pred))
        FN = sum((true == 1 and pred == 0) for true, pred in zip(y_true, y_pred))
        TN = sum((true == 0 and pred == 0) for true, pred in zip(y_true, y_pred))

        # Calculate metrics
        n = len(y_true)
        accuracy = (TP + TN) / n if n != 0 else 0.0

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

        metrics.append(("accuracy", accuracy))
        metrics.append(("f1", f1))
        metrics.append(("precision", precision))
        metrics.append(("recall", recall))

    elif system_type == "regression":
        # Calculate MAE and RMSE
        errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
        mae = sum(errors) / len(errors) if len(errors) != 0 else 0.0

        squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]
        rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5 if len(squared_errors) != 0 else 0.0

        metrics.append(("mae", mae))
        metrics.append(("rmse", rmse))

    elif system_type == "ranking":
        # Pair and sort by predicted score descending
        paired = list(zip(y_true, y_pred))
        paired.sort(key=lambda x: -x[1])

        # Calculate precision_at_3 and recall_at_3
        top_3 = paired[:3]
        relevant_in_top_3 = sum(1 for true, _ in top_3 if true == 1)
        total_relevant = sum(y_true)

        precision_at_3 = relevant_in_top_3 / 3 if 3 != 0 else 0.0
        recall_at_3 = relevant_in_top_3 / total_relevant if total_relevant != 0 else 0.0

        metrics.append(("precision_at_3", precision_at_3))
        metrics.append(("recall_at_3", recall_at_3))

    # Sort metrics alphabetically by name
    metrics.sort(key=lambda x: x[0])

    return metrics