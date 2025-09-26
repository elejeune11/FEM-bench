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
        raise ValueError('num_elements must be >= 1.')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3.')
    if not isinstance(material_regions, list) or len(material_regions) == 0:
        raise ValueError('material_regions must be a non-empty list.')
    if not isinstance(dirichlet_bc_list, list):
        raise ValueError('dirichlet_bc_list must be a list.')
    L = x_max - x_min
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes, dtype=float)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    if n_gauss == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif n_gauss == 2:
        v = 1.0 / np.sqrt(3.0)
        gauss_pts = np.array([-v, v])
        gauss_wts = np.array([1.0, 1.0])
    else:
        v = np.sqrt(3.0 / 5.0)
        gauss_pts = np.array([-v, 0.0, v])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    tol = max(1e-12, 1e-10 * L)

    def EA_at_x(x: float) -> float:
        for reg in material_regions:
            xmin = reg.get('coord_min', None)
            xmax = reg.get('coord_max', None)
            E = reg.get('E', None)
            A = reg.get('A', None)
            if xmin is None or xmax is None or E is None or (A is None):
                raise ValueError("Each material region must provide 'coord_min', 'coord_max', 'E', and 'A'.")
            if x >= xmin - tol and x <= xmax + tol:
                return float(E) * float(A)
        raise ValueError(f'Material properties not defined at x={x} within provided regions.')
    for e in range(num_elements):
        n1 = e
        n2 = e + 1
        x1 = node_coords[n1]
        x2 = node_coords[n2]
        Le = x2 - x1
        if not Le > 0.0:
            raise ValueError('Non-positive element length encountered.')
        J = 0.5 * Le
        B = np.array([-1.0 / Le, 1.0 / Le])
        Ke = np.zeros((2, 2), dtype=float)
        fe = np.zeros(2, dtype=float)
        x_mid = 0.5 * (x1 + x2)
        for (gp, w) in zip(gauss_pts, gauss_wts):
            xg = x_mid + J * gp
            kappa = EA_at_x(xg)
            N = np.array([0.5 * (1.0 - gp), 0.5 * (1.0 + gp)], dtype=float)
            Ke += w * J * kappa * np.outer(B, B)
            qx = float(body_force_fn(xg))
            fe += w * J * qx * N
        K[n1, n1] += Ke[0, 0]
        K[n1, n2] += Ke[0, 1]
        K[n2, n1] += Ke[1, 0]
        K[n2, n2] += Ke[1, 1]
        F[n1] += fe[0]
        F[n2] += fe[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            xloc = bc.get('x_location', None)
            load = bc.get('load_mag', None)
            if xloc is None or load is None:
                raise ValueError("Each Neumann BC must contain 'x_location' and 'load_mag'.")
            idx = int(np.argmin(np.abs(node_coords - xloc)))
            if not np.isclose(node_coords[idx], xloc, rtol=0.0, atol=tol):
                raise ValueError(f'Neumann BC location x={xloc} does not match any node coordinate.')
            F[idx] += float(load)
    if len(dirichlet_bc_list) == 0:
        raise ValueError('At least one Dirichlet BC required to prevent rigid body motion.')
    bc_map = {}
    for bc in dirichlet_bc_list:
        xloc = bc.get('x_location', None)
        up = bc.get('u_prescribed', None)
        if xloc is None or up is None:
            raise ValueError("Each Dirichlet BC must contain 'x_location' and 'u_prescribed'.")
        idx = int(np.argmin(np.abs(node_coords - xloc)))
        if not np.isclose(node_coords[idx], xloc, rtol=0.0, atol=tol):
            raise ValueError(f'Dirichlet BC location x={xloc} does not match any node coordinate.')
        if idx in bc_map:
            if not np.isclose(bc_map[idx], up, rtol=0.0, atol=1e-12):
                raise ValueError(f'Conflicting Dirichlet BCs at node index {idx}.')
        else:
            bc_map[idx] = float(up)
    reaction_nodes = np.array(sorted(bc_map.keys()), dtype=int)
    u_prescribed = np.array([bc_map[i] for i in reaction_nodes], dtype=float)
    all_idx = np.arange(n_nodes, dtype=int)
    free_mask = np.ones(n_nodes, dtype=bool)
    free_mask[reaction_nodes] = False
    free_nodes = all_idx[free_mask]
    u = np.zeros(n_nodes, dtype=float)
    if free_nodes.size == 0:
        u[reaction_nodes] = u_prescribed
    else:
        K_ff = K[np.ix_(free_nodes, free_nodes)]
        K_fc = K[np.ix](free_nodes, reaction_nodes)
        K_fc = K[np.ix_(free_nodes, reaction_nodes)]
        F_f = F[free_nodes]
        rhs = F_f - K_fc @ u_prescribed
        try:
            u_f = np.linalg.solve(K_ff, rhs)
        except np.linalg.LinAlgError as e:
            raise ValueError('Linear system is singular or ill-conditioned.') from e
        u[free_nodes] = u_f
        u[reaction_nodes] = u_prescribed
    reactions = K[np.ix_(reaction_nodes, all_idx)] @ u - F[reaction_nodes]
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}