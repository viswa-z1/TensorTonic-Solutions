import math

def evaluate_shadow(production_log, shadow_log, criteria):
    n = len(production_log)

    # Calculate accuracies
    production_correct = sum(1 for log in production_log if log["prediction"] == log["actual"])
    shadow_correct = sum(1 for log in shadow_log if log["prediction"] == log["actual"])

    production_accuracy = production_correct / n
    shadow_accuracy = shadow_correct / n
    accuracy_gain = shadow_accuracy - production_accuracy

    # Calculate P95 latency for shadow model
    shadow_latencies = [log["latency_ms"] for log in shadow_log]
    shadow_latencies_sorted = sorted(shadow_latencies)
    p95_index = math.ceil(0.95 * n) - 1
    shadow_latency_p95 = shadow_latencies_sorted[p95_index]

    # Calculate agreement rate
    agreement_count = sum(
        1 for p_log, s_log in zip(production_log, shadow_log)
        if p_log["prediction"] == s_log["prediction"]
    )
    agreement_rate = agreement_count / n

    # Prepare metrics
    metrics = {
        "shadow_accuracy": shadow_accuracy,
        "production_accuracy": production_accuracy,
        "accuracy_gain": accuracy_gain,
        "shadow_latency_p95": shadow_latency_p95,
        "agreement_rate": agreement_rate,
    }

    # Check promotion criteria
    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"] and
        shadow_latency_p95 <= criteria["max_latency_p95"] and
        agreement_rate >= criteria["min_agreement_rate"]
    )

    return {"promote": promote, "metrics": metrics}