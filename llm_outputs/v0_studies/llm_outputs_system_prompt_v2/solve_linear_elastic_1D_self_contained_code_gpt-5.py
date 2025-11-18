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
    import numpy as np
    if not x_max > x_min:
        raise ValueError('x_max must be greater than x_min.')
    if num_elements < 1:
        raise ValueError('num_elements must be at least 1.')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    L = x_max - x_min
    tol = max(1e-12, 1e-10 * L)
    le_all = node_coords[1:] - node_coords[:-1]
    mids = 0.5 * (node_coords[:-1] + node_coords[1:])
    E_e = np.empty(num_elements, dtype=float)
    A_e = np.empty(num_elements, dtype=float)
    for e in range(num_elements):
        x_mid = mids[e]
        selected = None
        for r in material_regions:
            rmin = float(r['coord_min'])
            rmax = float(r['coord_max'])
            if x_mid >= rmin - tol and x_mid <= rmax + tol:
                selected = r
                break
        if selected is None:
            raise ValueError(f'No material region covers element with midpoint {x_mid}.')
        E_e[e] = float(selected['E'])
        A_e[e] = float(selected['A'])
    if n_gauss == 1:
        xi = np.array([0.0])
        w = np.array([2.0])
    elif n_gauss == 2:
        xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w = np.array([1.0, 1.0])
    else:
        xi = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        i = e
        j = e + 1
        x1 = node_coords[i]
        x2 = node_coords[j]
        le = le_all[e]
        E = E_e[e]
        A = A_e[e]
        ke = E * A / le * np.array([[1.0, -1.0], [-1.0, 1.0]])
        fe = np.zeros(2, dtype=float)
        J = le / 2.0
        for g in range(len(xi)):
            xi_g = xi[g]
            wg = w[g]
            N1 = 0.5 * (1.0 - xi_g)
            N2 = 0.5 * (1.0 + xi_g)
            xg = N1 * x1 + N2 * x2
            fg = float(body_force_fn(float(xg)))
            fe[0] += N1 * fg * J * wg
            fe[1] += N2 * fg * J * wg
        K[i, i] += ke[0, 0]
        K[i, j] += ke[0, 1]
        K[j, i] += ke[1, 0]
        K[j, j] += ke[1, 1]
        F[i] += fe[0]
        F[j] += fe[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = float(bc['x_location'])
            load = float(bc['load_mag'])
            idx = int(np.argmin(np.abs(node_coords - x_loc)))
            if abs(node_coords[idx] - x_loc) > tol:
                raise ValueError(f'Neumann BC location {x_loc} does not correspond to a mesh node.')
            F[idx] += load
    K_orig = K.copy()
    F_total = F.copy()
    if dirichlet_bc_list is None or len(dirichlet_bc_list) == 0:
        raise ValueError('At least one Dirichlet boundary condition is required to obtain a unique solution.')
    dir_nodes_list = []
    dir_values_list = []
    used = {}
    for bc in dirichlet_bc_list:
        x_loc = float(bc['x_location'])
        u_val = float(bc['u_prescribed'])
        idx = int(np.argmin(np.abs(node_coords - x_loc)))
        if abs(node_coords[idx] - x_loc) > tol:
            raise ValueError(f'Dirichlet BC location {x_loc} does not correspond to a mesh node.')
        if idx in used:
            if not np.isclose(used[idx], u_val):
                raise ValueError(f'Conflicting Dirichlet BCs at node {idx}.')
            else:
                raise ValueError(f'Duplicate Dirichlet BC specified at node {idx}.')
        used[idx] = u_val
        dir_nodes_list.append(idx)
        dir_values_list.append(u_val)
    dir_nodes = np.array(dir_nodes_list, dtype=int)
    dir_values = np.array(dir_values_list, dtype=float)
    K_mod = K.copy()
    F_mod = F.copy()
    for (idx, u_val) in zip(dir_nodes, dir_values):
        F_mod -= K_mod[:, idx] * u_val
    for (idx, u_val) in zip(dir_nodes, dir_values):
        K_mod[:, idx] = 0.0
        K_mod[idx, :] = 0.0
        K_mod[idx, idx] = 1.0
        F_mod[idx] = u_val
    displacements = np.linalg.solve(K_mod, F_mod)
    R_full = K_orig @ displacements - F_total
    reactions = R_full[dir_nodes] if dir_nodes.size > 0 else np.array([], dtype=float)
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dir_nodes}