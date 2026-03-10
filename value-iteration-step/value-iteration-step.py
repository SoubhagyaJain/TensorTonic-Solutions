def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    new_values = []

    for s in range(len(values)):
        best_q = float("-inf")

        for a in range(len(transitions[s])):
            expected_future = 0.0
            for s_next in range(len(values)):
                expected_future += transitions[s][a][s_next] * values[s_next]

            q_value = rewards[s][a] + gamma * expected_future
            best_q = max(best_q, q_value)

        new_values.append(float(best_q))

    return new_values