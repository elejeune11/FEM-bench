def solve_linear_elastic_1D_self_contained(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    n_nodes = len(node_coords)
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(0.6), 0, np.sqrt(0.6)])
        gauss_weights = np.array([5 / 9, 8 / 9, 5 / 9])
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    for e in range(num_elements):
        (x1, x2) = (node_coords[e], node_coords[e + 1])
        Le = x2 - x1
        E = A = None
        for region in material_regions:
            if (x1 + x2) / 2 >= region['coord_min'] and (x1 + x2) / 2 <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material properties found for element at x = {(x1 + x2) / 2}')
        Ke = np.array([[1, -1], [-1, 1]]) * (E * A / Le)
        Fe = np.zeros(2)
        for (i, xi) in enumerate(gauss_points):
            x = x1 + (1 + xi) * Le / 2
            N1 = (1 - xi) / 2
            N2 = (1 + xi) / 2
            Fe += gauss_weights[i] * np.array([N1, N2]) * body_force_fn(x) * Le / 2
        dofs = [e, e + 1]
        K[np.ix_(dofs, dofs)] += Ke
        F[dofs] += Fe
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            node_idx = np.abs(node_coords - bc['x_location']).argmin()
            F[node_idx] += bc['load_mag']
    free_dofs = np.ones(n_nodes, dtype=bool)
    prescribed_values = np.zeros(n_nodes)
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        node_idx = np.abs(node_coords - bc['x_location']).argmin()
        free_dofs[node_idx] = False
        prescribed_values[node_idx] = bc['u_prescribed']
        reaction_nodes.append(node_idx)
    reaction_nodes = np.array(reaction_nodes)
    constrained_dofs = ~free_dofs
    F[free_dofs] -= K[np.ix_(free_dofs, constrained_dofs)] @ prescribed_values[constrained_dofs]
    u = prescribed_values.copy()
    u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])
    reactions = K @ u - F
    reactions = reactions[reaction_nodes]
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}