import numpy as np


def apply_cost_function(data, fun):
    """ Calculate tranitions costs in between states. """
    res = []
    for i in range(1, len(data)):
        res.append(fun(data[i - 1], data[i]))
    return res


def calculate_value_function(transition_costs):
    """Recursively apply the bellman equation from the end to the start. """
    state_dim = [tc.shape[0] for tc in transition_costs]
    state_dim.append(transition_costs[-1].shape[1])

    V = [np.zeros(d) for d in state_dim]
    V_ind = [np.zeros(d) for d in state_dim]

    for i in range(len(state_dim) - 2, -1, -1):
        rhs = transition_costs[i] + V[i + 1]
        V[i] = np.min(rhs, axis=1)
        V_ind[i] = np.argmin(rhs, axis=1)

    return V_ind, V


def extract_shortest_path(data, V_ind, V):
    """ Start in stage one with the state that has to lowest value in V
    and follow the index trace from then on.
    """
    res = []
    i_start = np.argmin(V[0])
    i_next = i_start
    for i_stage in range(len(data)):
        res.append(data[i_stage][i_next])
        i_next = V_ind[i_stage][i_next]

    return res
