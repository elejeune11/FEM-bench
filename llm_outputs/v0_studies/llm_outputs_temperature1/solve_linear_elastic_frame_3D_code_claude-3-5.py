def solve_linear_elastic_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    n_nodes = len(node_coords)
    bc_bool = {}
    for (node, flags) in boundary_conditions.items():
        bc_bool[node] = [bool(f) for f in flags]
    K = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    (fixed, free) = partition_degrees_of_freedom(bc_bool, n_nodes)
    (u, r) = linear_solve(P, K, fixed, free)
    return (u, r)