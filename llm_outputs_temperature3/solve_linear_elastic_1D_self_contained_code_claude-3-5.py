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
        gauss_points = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
        gauss_weights = np.array([5 / 9, 8 / 9, 5 / 9])
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    for e in range(num_elements):
        (x1, x2) = (node_coords[e], node_coords[e + 1])
        Le = x2 - x1
        E = A = None
        for region in material_regions:
            if x1 >= region['coord_min'] and x2 <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material properties found for element {e}')
        ke = np.zeros((2, 2))
        fe = np.zeros(2)
        for i in range(n_gauss):
            xi = gauss_points[i]
            w = gauss_weights[i]
            dN = np.array([-1 / 2, 1 / 2])
            J = Le / 2
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            x = x1 + (1 + xi) * Le / 2
            ke += E * A / J * np.outer(dN, dN) * w * J
            fe += N * body_force_fn(x) * w * J
        nodes = [e, e + 1]
        for i in range(2):
            for j in range(2):
                K[nodes[i], nodes[j]] += ke[i, j]
            F[nodes[i]] += fe[i]
    if neumann_bc_list:
        for bc in neumann_bc_list:
            node_idx = np.abs(node_coords - bc['x_location']).argmin()
            F[node_idx] += bc['load_mag']
    free_dofs = list(range(n_nodes))
    prescribed_dofs = []
    prescribed_values = []
    for bc in dirichlet_bc_list:
        node_idx = np.abs(node_coords - bc['x_location']).argmin()
        prescribed_dofs.append(node_idx)
        prescribed_values.append(bc['u_prescribed'])
        free_dofs.remove(node_idx)
    u = np.zeros(n_nodes)
    u[prescribed_dofs] = prescribed_values
    K_free = K[np.ix_(free_dofs, free_dofs)]
    F_free = F[free_dofs] - K[np.ix_(free_dofs, prescribed_dofs)] @ prescribed_values
    u[free_dofs] = np.linalg.solve(K_free, F_free)
    reactions = K @ u - F
    return {'displacements': u, 'reactions': reactions[prescribed_dofs], 'node_coords': node_coords, 'reaction_nodes': np.array(prescribed_dofs)}