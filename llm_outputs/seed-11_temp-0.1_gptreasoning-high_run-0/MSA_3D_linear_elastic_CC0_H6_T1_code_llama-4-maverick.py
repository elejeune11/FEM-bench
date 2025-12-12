def MSA_3D_linear_elastic_CC0_H6_T1(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = node_coords.shape[0]
    K_global = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P_global = assemble_global_load_vector_linear_elastic_3D({n: load for (n, load) in nodal_loads.items()}, n_nodes)
    bc_bool = {n: [bool(f) for f in flags] for (n, flags) in boundary_conditions.items()}
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u, r) = linear_solve(P_global, K_global, fixed, free)
    return (u, r)