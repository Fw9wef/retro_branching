import numpy as np
from numpy import ndarray
from numpy.random import default_rng

import pyscipopt as scip
from pyscipopt import quicksum

def floyd_warshall(x: ndarray) -> ndarray:
    """Floyd-Warshall all-pairs shortest paths to enforce triangle inequality."""
    *ignore, n = x.shape
    x = x.copy()

    # zero the diagonal
    i, j = np.diag_indices(n)
    x[..., i, j] = 0

    # d^{k+1}_{uv} = \min\{d^k_{uv}, d^k_{uk} + d^k_{kv}\}
    b = np.empty_like(x)
    for k in range(n):
        np.minimum(x, np.add.outer(x[..., :, k], x[..., k, :], out=b), out=x)

    return x

def tsp_generate(
    n_nodes: int = 20,
    *,
    seed: int = None,
    planar: bool = False,
) -> ndarray:
    """Simple TWCVRP generator."""
    rng = default_rng(seed)

    # generate some distances, make times equal distances
    if planar:
        xy = rng.uniform(size=(n_nodes, 2))
        T = np.linalg.norm(xy[np.newaxis, :] - xy[:, np.newaxis], axis=-1)

    else:
        xy = None
        T = floyd_warshall(-np.log(rng.uniform(size=(n_nodes, n_nodes))))

    return T


def tsp_setup(T: ndarray) -> scip.Model:
    """A TSP problem on the distance-like cost matrix T using MTZ formulation."""
    m = scip.Model()
    N, _ = T.shape
    nodes = range(N)

    # x_{uv} -- arc u-v is on the route
    # t_u -- time of visit of a route starting at the zero-th node
    x, t, alpha, omega = {}, {}, {}, {}
    for u in nodes:
        for v in nodes:
            x[u, v] = m.addVar(f"x[{u}, {v}]", "B")

    # depart-arrive expressions (alpha-omega)
    for v in nodes:
        alpha[v] = quicksum(x[v, u] for u in nodes)  # \alpha_v
        omega[v] = quicksum(x[u, v] for u in nodes)  # \omega_v

    # inflow-outflow balance: \alpha_v = \omega_v
    for v in nodes:
        m.addCons(alpha[v] == omega[v], f"flow[{v}]")

    # \alpha_v = 1  % must depart location v exactly once
    for v in nodes:
        m.addCons(alpha[v] == 1, f"once[{v}]")

    # visit-order monotonic integral time variables for MTZ
    for v in nodes:
        t[v] = m.addVar(f"t[{v}]", "I", lb=0, ub=N)

    # loops are fobidden automatically, but MTZ requires that one node be skipped
    #  this is ok, since every node has to be visited.
    for u in nodes:
        if u == 0:
            continue

        for v in nodes:
            m_uv = t[u] + 1 - t[v]
            m.addCons(m_uv <= (1 - x[u, v]) * (N + 1), f"mtz[{u}, {v}]")

    # the objective is to minimize the total travel cost \sum_{uv} C_{uv} x_{uv}
    m.setObjective(
        quicksum(x[u, v] * T[u, v] for u in nodes for v in nodes),
        "minimize",
    )

    m.data = x, t, N

    return m


import ecole
from ecole.scip import Model

def tsp_instance_generator(
    n_nodes: int = 35,
    *,
    seed: int = None,
    planar: bool = False,
) -> Model:
    # pip install git+https://github.com/ivannz/branching-crustaceans.git
    while True:
        instance = tsp_generate(n_nodes=n_nodes, seed=seed, planar=planar)
        yield Model.from_pyscipopt(tsp_setup(instance))
