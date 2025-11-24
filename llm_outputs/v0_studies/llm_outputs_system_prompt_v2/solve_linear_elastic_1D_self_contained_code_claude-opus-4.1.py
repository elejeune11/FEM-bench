def solve_linear_elastic_1D_self_contained(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    """
    Solve a 1D linear elastic finite element problem with integrated meshing.
    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.
        material_regions (List[Dict]): List of material regions, each with keys:
            "coord_min", "coord_max", "E", "A".
        body_force_fn (Callable): Function f(x) for body force.
        dirichlet_bc_list (List[Dict]): Each dict must contain:
            {
                "x_location": float,      # coordinate of prescribed node
                "u_prescribed": float     # displacement value
            }
        neumann_bc_list (Optional[List[Dict]]): Each dict must contain:
            {
                "x_location": float,  # coordinate of the node
                "load_mag": float     # magnitude of point load (positive = outward)
            }
        n_gauss (int): Number of Gauss points for numerical integration (1 to 3 supported).
    Returns:
        dict: Dictionary containing solution results:
    """
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    gauss_points_xi = {1: np.array([0.0]), 2: np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]), 3: np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])}
    gauss_weights = {1: np.array([2.0]), 2: np.array([1.0, 1.0]), 3: np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])}
    xi_pts = gauss_points_xi[n_gauss]
    w_pts = gauss_weights[n_gauss]
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        h_e = x2 - x1
        x_mid = 0.5 * (x1 + x2)
        E_val = 0.0
        A_val = 0.0
        for region in material_regions:
            if region['coord_min'] <= x_mid <= region['coord_max']:
                E_val = region['E']
                A_val = region['A']
                break
        k_e = np.zeros((2, 2))
        f_e = np.zeros(2)
        for (i, xi) in enumerate(xi_pts):
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            dN1_dxi = -0.5
            dN2_dxi = 0.5
            J = 0.5 * h_e
            dN1_dx = dN1_dxi / J
            dN2_dx = dN2_dxi / J
            B = np.array([dN1_dx, dN2_dx])
            k_e += w_pts[i] * E_val * A_val * np.outer(B, B) * J
            x_phys = N1 * x1 + N2 * x2
            f_val = body_force_fn(x_phys)
            f_e[0] += w_pts[i] * f_val * N1 * J
            f_e[1] += w_pts[i] * f_val * N2 * J
        global_dofs = [e, e + 1]
        for i in range(2):
            F_global[global_dofs[i]] += f_e[i]
            for j in range(2):
                K_global[global_dofs[i], global_dofs[j]] += k_e[i, j]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
            F_global[node_idx] += bc['load_mag']
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(bc['u_prescribed'])
    dirichlet_nodes = np.array(dirichlet_nodes, dtype=int)
    dirichlet_values = np.array(dirichlet_values)
    free_dofs = np.setdiff1d(np.arange(n_nodes), dirichlet_nodes)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fc = K_global[np.ix_(free_dofs, dirichlet_nodes)]
    F_f = F_global[free_dofs]
    F_f_mod = F_f - K_fc @ dirichlet_values
    u_free = np.linalg.solve(K_ff, F_f_mod)
    u_full = np.zeros(n_nodes)
    u_full[dirichlet_nodes] = dirichlet_values
    u_full[free_dofs] = u_free
    reactions = K_global[dirichlet_nodes, :] @ u_full - F_global[dirichlet_nodes]
    return {'displacements': u_full, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}