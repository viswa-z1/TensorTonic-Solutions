def retraining_policy(daily_stats, config):
    retrain_days = []
    days_since_retrain = 0
    remaining_budget = config["budget"]
    last_retrain_day = -config["cooldown"]  # Initialize to satisfy cooldown on day 1

    for day_stat in daily_stats:
        day = day_stat["day"]
        drift_score = day_stat["drift_score"]
        performance = day_stat["performance"]

        # Increment days since last retrain
        days_since_retrain += 1

        # Check trigger conditions
        drift_trigger = drift_score > config["drift_threshold"]
        performance_trigger = performance < config["performance_threshold"]
        staleness_trigger = days_since_retrain >= config["max_staleness"]

        # Check constraints
        cooldown_satisfied = (day - last_retrain_day) >= config["cooldown"]
        budget_satisfied = remaining_budget >= config["retrain_cost"]

        # Decide if retraining should be triggered
        if (drift_trigger or performance_trigger or staleness_trigger) and cooldown_satisfied and budget_satisfied:
            retrain_days.append(day)
            days_since_retrain = 0
            remaining_budget -= config["retrain_cost"]
            last_retrain_day = day

    return retrain_days