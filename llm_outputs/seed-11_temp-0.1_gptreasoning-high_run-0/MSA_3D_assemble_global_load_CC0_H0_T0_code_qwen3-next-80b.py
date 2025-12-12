def MSA_3D_assemble_global_load_CC0_H0_T0(nodal_loads: dict, n_nodes: int):
    P = np.zeros(6 * n_nodes)
    for (node_idx, load_vector) in nodal_loads.items():
        start_idx = 6 * node_idx
        P[start_idx:start_idx + 6] = np.array(load_vector, dtype=float)
    return P