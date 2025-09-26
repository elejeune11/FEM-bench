def solve_linear_elastic_1D_self_contained(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    if n_gauss == 1:
        gauss_points = [0.0]
        gauss_weights = [2.0]
    elif n_gauss == 2:
        gauss_points = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_weights = [1.0, 1.0]
    elif n_gauss == 3:
        gauss_points = [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]
        gauss_weights = [5 / 9, 8 / 9, 5 / 9]
    else:
        raise ValueError('n_gauss must be 1, 2, or 3')
    for elem_idx in range(num_elements):
        node1 = elem_idx
        node2 = elem_idx + 1
        x1 = node_coords[node1]
        x2 = node_coords[node2]
        L = x2 - x1
        elem_center = (x1 + x2) / 2
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= elem_center <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        if E is None or A is None:
            raise ValueError(f'No material region found for element centered at {elem_center}')
        k_elem = E * A / L * np.array([[1, -1], [-1, 1]])
        f_elem = np.zeros(2)
        for (gp, weight) in zip(gauss_points, gauss_weights):
            N1 = 0.5 * (1 - gp)
            N2 = 0.5 * (1 + gp)
            J = L / 2
            x_gp = N1 * x1 + N2 * x2
            b_gp = body_force_fn(x_gp)
            f_elem[0] += weight * b_gp * N1 * J
            f_elem[1] += weight * b_gp * N2 * J
        K_global[node1:node2 + 1, node1:node2 + 1] += k_elem
        F_global[node1:node2 + 1] += f_elem
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load
    dirichlet_nodes = []
    u_prescribed = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_pres = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        u_prescribed.append(u_pres)
    dirichlet_nodes = np.array(dirichlet_nodes)
    u_prescribed = np.array(u_prescribed)
    free_dofs = np.setdiff1d(np.arange(n_nodes), dirichlet_nodes)
    fixed_dofs = dirichlet_nodes
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fc = K_global[np.ix_(free_dofs, fixed_dofs)]
    F_f = F_global[free_dofs]
    u_free = np.linalg.solve(K_ff, F_f - K_fc @ u_prescribed)
    displacements = np.zeros(n_nodes)
    displacements[free_dofs] = u_free
    displacements[fixed_dofs] = u_prescribed
    reactions = K_global @ displacements - F_global
    reactions = reactions[fixed_dofs]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}