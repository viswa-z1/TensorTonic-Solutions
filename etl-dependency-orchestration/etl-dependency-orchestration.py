def schedule_pipeline(tasks, resource_budget):
    from collections import defaultdict, deque

    # Initialize task information
    task_info = {task['name']: task for task in tasks}
    dependency_counts = {task['name']: len(task['depends_on']) for task in tasks}
    reverse_dependencies = defaultdict(list)
    for task in tasks:
        for dep in task['depends_on']:
            reverse_dependencies[dep].append(task['name'])

    # Initialize ready queue and running tasks
    ready_queue = []
    for task in tasks:
        if dependency_counts[task['name']] == 0:
            ready_queue.append(task['name'])
    ready_queue.sort()  # Sort alphabetically

    running_tasks = {}  # {task_name: end_time}
    current_time = 0
    schedule = []
    completed_tasks = set()

    while ready_queue or running_tasks:
        # Complete tasks that have finished by current_time
        finished_tasks = [task for task, end_time in running_tasks.items() if end_time <= current_time]
        for task in finished_tasks:
            del running_tasks[task]
            completed_tasks.add(task)
            # Update dependency counts for tasks that depend on this one
            for dependent in reverse_dependencies[task]:
                dependency_counts[dependent] -= 1
                if dependency_counts[dependent] == 0 and dependent not in completed_tasks and dependent not in running_tasks:
                    ready_queue.append(dependent)
        ready_queue.sort()  # Re-sort after adding new tasks

        # Assign as many ready tasks as possible without exceeding the budget
        used_resources = sum(task_info[task]['resources'] for task in running_tasks)
        i = 0
        while i < len(ready_queue):
            task_name = ready_queue[i]
            task_resources = task_info[task_name]['resources']
            if used_resources + task_resources <= resource_budget:
                start_time = current_time
                end_time = start_time + task_info[task_name]['duration']
                running_tasks[task_name] = end_time
                schedule.append((task_name, start_time))
                used_resources += task_resources
                ready_queue.pop(i)
            else:
                i += 1

        # Advance time to the next completion event
        if running_tasks:
            next_completion_time = min(running_tasks.values())
            current_time = next_completion_time
        else:
            # No running tasks, but ready_queue is not empty (shouldn't happen as per constraints)
            current_time += 1

    # Sort the schedule by start_time and then by task_name
    schedule.sort(key=lambda x: (x[1], x[0]))
    return schedule