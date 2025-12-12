def FEM_1D_linear_elastic_CC0_H0_T0(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    if n_gauss not in {1, 2, 3}:
        raise ValueError('n_gauss must be 1, 2, or 3')
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    n_nodes = len(node_coords)
    element_E = np.zeros(num_elements)
    element_A = np.zeros(num_elements)
    for i in range(num_elements):
        x_center = (node_coords[i] + node_coords[i + 1]) / 2.0
        found = False
        for region in material_regions:
            if region['coord_min'] <= x_center < region['coord_max']:
                element_E[i] = region['E']
                element_A[i] = region['A']
                found = True
                break
        if not found:
            raise ValueError('Element not covered by exactly one material region')
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    if n_gauss == 1:
        xi_gauss = [0.0]
        w_gauss = [2.0]
    elif n_gauss == 2:
        xi_gauss = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        w_gauss = [1.0, 1.0]
    else:
        xi_gauss = [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]
        w_gauss = [5 / 9, 8 / 9, 5 / 9]
    for elem in range(num_elements):
        (x1, x2) = (node_coords[elem], node_coords[elem + 1])
        L = x2 - x1
        E = element_E[elem]
        A = element_A[elem]
        EA = E * A
        K_elem = EA / L * np.array([[1, -1], [-1, 1]])
        K_global[elem, elem] += K_elem[0, 0]
        K_global[elem, elem + 1] += K_elem[0, 1]
        K_global[elem + 1, elem] += K_elem[1, 0]
        K_global[elem + 1, elem + 1] += K_elem[1, 1]
        for i in range(n_gauss):
            xi = xi_gauss[i]
            w = w_gauss[i]
            x = (1 - xi) * x1 / 2 + (1 + xi) * x2 / 2
            f_x = body_force_fn(x)
            N1 = (1 - xi) / 2
            N2 = (1 + xi) / 2
            F_global[elem] += f_x * N1 * (L / 2) * w
            F_global[elem + 1] += f_x * N2 * (L / 2) * w
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            if abs(node_coords[node_idx] - x_loc) > 1e-10:
                raise ValueError('Neumann BC coordinate does not match a node')
            F_global[node_idx] += load_mag
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        if abs(node_coords[node_idx] - x_loc) > 1e-10:
            raise ValueError('Dirichlet BC coordinate does not match a node')
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(u_prescribed)
    fixed_nodes = set(dirichlet_nodes)
    free_nodes = [i for i in range(n_nodes) if i not in fixed_nodes]
    n_free = len(free_nodes)
    n_dirichlet = len(dirichlet_nodes)
    if n_free == 0:
        raise np.linalg.LinAlgError('Insufficient Dirichlet constraints')
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_free = F_global[free_nodes] - K_fd @ np.array(dirichlet_values)
    u_free = np.linalg.solve(K_ff, F_free)
    u_global = np.zeros(n_nodes)
    u_global[free_nodes] = u_free
    u_global[dirichlet_nodes] = dirichlet_values
    K_df = K_global[np.ix_(dirichlet_nodes, free_nodes)]
    K_dd = K_global[np.ix_(dirichlet_nodes, dirichlet_nodes)]
    reactions = K_df @ u_free + K_dd @ np.array(dirichlet_values) - F_global[dirichlet_nodes]
    return {'displacements': u_global, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}