def MSA_3D_linear_elastic_CC0_H6_T1(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    import numpy as np
    n_nodes = np.asarray(node_coords).shape[0]
    bc_bool_map = {}
    for n, flags in (boundary_conditions or {}).items():
        bc_bool_map[int(n)] = np.asarray(flags, dtype=bool)
    loads_float_map = {}
    for n, load in (nodal_loads or {}).items():
        loads_float_map[int(n)] = np.asarray(load, dtype=float)
    K_global = assemble_global_stiffness_matrix_linear_elastic_3D(np.asarray(node_coords, dtype=float), elements)
    P_global = assemble_global_load_vector_linear_elastic_3D(loads_float_map, n_nodes)
    fixed, free = partition_degrees_of_freedom(bc_bool_map, n_nodes)
    n_dof = 6 * n_nodes
    if free.size == 0:
        u = np.zeros(n_dof)
        r = np.zeros(n_dof)
        r[fixed] = -P_global[fixed]
        return (u, r)
    u, r = linear_solve(P_global, K_global, fixed, free)
    return (u, r)