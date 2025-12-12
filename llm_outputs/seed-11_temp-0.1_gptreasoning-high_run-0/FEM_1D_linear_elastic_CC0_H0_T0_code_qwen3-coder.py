def FEM_1D_linear_elastic_CC0_H0_T0(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if num_elements < 1:
        raise ValueError('num_elements must be at least 1.')
    if n_gauss not in [1, 2, 3]:
        raise ValueError('n_gauss must be 1, 2, or 3.')
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_length = (x_max - x_min) / num_elements
    if n_gauss == 1:
        xi_g = np.array([0.0])
        w_g = np.array([2.0])
    elif n_gauss == 2:
        xi_g = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        w_g = np.array([1.0, 1.0])
    elif n_gauss == 3:
        xi_g = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        w_g = np.array([5 / 9, 8 / 9, 5 / 9])
    E_vec = np.zeros(num_elements)
    A_vec = np.zeros(num_elements)
    for i in range(num_elements):
        x_center = x_min + (i + 0.5) * element_length
        region_found = False
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                if region_found:
                    raise ValueError(f'Element {i} midpoint at {x_center} is covered by multiple regions.')
                E_vec[i] = region['E']
                A_vec[i] = region['A']
                region_found = True
        if not region_found:
            raise ValueError(f'Element {i} midpoint at {x_center} is not covered by any region.')
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
    K_elem = np.array([[1.0, -1.0], [-1.0, 1.0]])
    for i in range(num_elements):
        (E, A, L) = (E_vec[i], A_vec[i], element_length)
        k_e = E * A / L * K_elem
        f_idx = [i, i + 1]
        K_global[np.ix_(f_idx, f_idx)] += k_e
        (x0, x1) = (node_coords[i], node_coords[i + 1])
        J = L / 2
        for (xi, w) in zip(xi_g, w_g):
            x_phys = (x0 + x1) / 2 + (x1 - x0) / 2 * xi
            f_val = body_force_fn(x_phys)
            N1 = (1 - xi) / 2
            N2 = (1 + xi) / 2
            f_e = f_val * J * w * np.array([N1, N2])
            F_global[f_idx] += f_e
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            idx = np.argmin(np.abs(node_coords - x_loc))
            if abs(node_coords[idx] - x_loc) > 1e-10:
                raise ValueError(f'Neumann BC location {x_loc} does not match any node.')
            F_global[idx] += load
    dirichlet_nodes = []
    u_prescribed = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_val = bc['u_prescribed']
        idx = np.argmin(np.abs(node_coords - x_loc))
        if abs(node_coords[idx] - x_loc) > 1e-10:
            raise ValueError(f'Dirichlet BC location {x_loc} does not match any node.')
        dirichlet_nodes.append(idx)
        u_prescribed.append(u_val)
    dirichlet_nodes = np.array(dirichlet_nodes)
    u_prescribed = np.array(u_prescribed)
    free_nodes = np.setdiff1d(np.arange(num_nodes), dirichlet_nodes)
    if len(free_nodes) == 0:
        raise np.linalg.LinAlgError('All nodes are constrained; system is over-constrained.')
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fd = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    u_d = u_prescribed
    u_f = np.linalg.solve(K_ff, F_f - K_fd @ u_d)
    u_global = np.zeros(num_nodes)
    u_global[free_nodes] = u_f
    u_global[dirichlet_nodes] = u_d
    K_df = K_global[np.ix_(dirichlet_nodes, free_nodes)]
    K_dd = K_global[np.ix_(dirichlet_nodes, dirichlet_nodes)]
    F_d = F_global[dirichlet_nodes]
    R = K_df @ u_f + K_dd @ u_d - F_d
    return {'displacements': u_global, 'reactions': R, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}