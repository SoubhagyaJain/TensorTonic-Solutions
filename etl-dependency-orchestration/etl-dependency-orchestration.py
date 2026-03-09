def schedule_pipeline(tasks, resource_budget):
    """
    Schedule ETL tasks respecting dependencies and resource limits.

    Returns:
        List of (task_name, start_time) sorted by (start_time, task_name)
    """
    # Index tasks by name for O(1) access
    task_map = {task["name"]: task for task in tasks}

    completed = set()
    started = set()
    running = []   # list of (end_time, task_name, resources)
    schedule = []

    time = 0

    while len(started) < len(tasks):
        # 1) Complete all tasks finishing at current time
        still_running = []
        for end_time, name, resources in running:
            if end_time <= time:
                completed.add(name)
            else:
                still_running.append((end_time, name, resources))
        running = still_running

        # Current resource usage
        used_resources = sum(resources for _, _, resources in running)

        # 2) Find ready tasks
        ready = []
        for task in tasks:
            name = task["name"]
            if name in started:
                continue
            if all(dep in completed for dep in task["depends_on"]):
                ready.append(task)

        # 3) Sort ready tasks alphabetically and greedily start them
        ready.sort(key=lambda t: t["name"])

        for task in ready:
            name = task["name"]
            duration = task["duration"]
            resources = task["resources"]

            if used_resources + resources <= resource_budget:
                started.add(name)
                schedule.append((name, time))
                running.append((time + duration, name, resources))
                used_resources += resources

        # 4) Advance time to next completion event
        if len(started) == len(tasks):
            break

        if running:
            time = min(end_time for end_time, _, _ in running)
        else:
            # For a valid DAG with schedulable tasks, this should not happen,
            # but keep it safe.
            break

    return sorted(schedule, key=lambda x: (x[1], x[0]))