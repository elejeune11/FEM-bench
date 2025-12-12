def FEM_1D_linear_elastic_CC0_H0_T0(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    """
    Solve a 1D small-strain, linear-elastic bar problem with built-in meshing and assembly.
    Builds a uniform 2-node (linear) mesh on [x_min, x_max], assigns element-wise
    material properties from piecewise regions, assembles the global stiffness and
    force vectors (including body forces via Gauss quadrature and optional point
    Neumann loads), applies Dirichlet constraints, solves for nodal displacements,
    and returns displacements and reactions.
    Parameters
    ----------
    x_min, x_max : float
        Domain bounds (x_max > x_min).
    num_elements : int
        Number of linear elements (num_nodes = num_elements + 1).
    material_regions : list of dict
        Piecewise-constant material regions. Each dict must contain:
        {"coord_min": float, "coord_max": float, "E": float, "A": float}.
        Elements are assigned by their midpoint x_center. Every element must fall
        into exactly one region.
    body_force_fn : Callable[[float], float]
        Body force density f(x) (force per unit length). Evaluated at quadrature points.
    dirichlet_bc_list : list of dict
        Dirichlet boundary conditions applied at existing mesh nodes. Each dict:
        {"x_location": float, "u_prescribed": float}.
    neumann_bc_list : list of dict, optional
        Point loads applied at existing mesh nodes. Each dict:
        {"x_location": float, "load_mag": float}.
        Positive load acts in the +x direction (outward).
    n_gauss : int, optional
        Number of Gauss points per element (1–3 supported). Default 2 (exact for
        linear elements with constant EA and linear f mapped through x(ξ)).
    Returns
    -------
    dict
        {
            "displacements": np.ndarray,  shape (n_nodes,)
                Nodal displacement vector.
            "reactions": np.ndarray,      shape (n_dirichlet,)
                Reaction forces at the Dirichlet-constrained nodes (in the order
                listed by `dirichlet_bc_list` / discovered nodes).
            "node_coords": np.ndarray,    shape (n_nodes,)
                Coordinates of all mesh nodes.
            "reaction_nodes": np.ndarray, shape (n_dirichlet,)
                Indices of nodes where reactions are reported.
        }
    Raises
    ------
    ValueError
        If an element is not covered by exactly one material region, if BC
        coordinates do not match a unique node, or if `n_gauss` is not in {1,2,3}.
    numpy.linalg.LinAlgError
        If the reduced stiffness matrix is singular (e.g., insufficient Dirichlet constraints).
    """
    if not x_max > x_min:
        raise ValueError('x_max must be greater than x_min.')
    if num_elements < 1:
        raise ValueError('num_elements must be at least 1.')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be in {1, 2, 3}.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    L = x_max - x_min
    tol = 1e-12 * max(1.0, abs(L))
    if n_gauss == 1:
        gp = np.array([0.0])
        gw = np.array([2.0])
    elif n_gauss == 2:
        a = 1.0 / np.sqrt(3.0)
        gp = np.array([-a, a])
        gw = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        gp = np.array([-a, 0.0, a])
        gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    for e in range(num_elements):
        iL = e
        iR = e + 1
        xa = node_coords[iL]
        xb = node_coords[iR]
        h = xb - xa
        x_center = 0.5 * (xa + xb)
        matched_regions = []
        for reg in material_regions:
            cmin = float(reg['coord_min'])
            cmax = float(reg['coord_max'])
            if x_center >= cmin and x_center <= cmax:
                matched_regions.append(reg)
        if len(matched_regions) != 1:
            raise ValueError('Each element must be covered by exactly one material region.')
        E = float(matched_regions[0]['E'])
        A = float(matched_regions[0]['A'])
        ke = E * A / h
        K[iL, iL] += ke
        K[iL, iR] -= ke
        K[iR, iL] -= ke
        K[iR, iR] += ke
        J = h * 0.5
        fe0 = 0.0
        fe1 = 0.0
        for (xi, w) in zip(gp, gw):
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            xg = N1 * xa + N2 * xb
            fg = float(body_force_fn(float(xg)))
            fe0 += w * N1 * fg * J
            fe1 += w * N2 * fg * J
        F[iL] += fe0
        F[iR] += fe1
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = float(bc['x_location'])
            load = float(bc['load_mag'])
            d = np.abs(node_coords - x_loc)
            idx = int(np.argmin(d))
            if d[idx] > tol:
                raise ValueError('Neumann BC x_location does not match a unique node.')
            F[idx] += load
    constrained_map = {}
    constrained_order = []
    for bc in dirichlet_bc_list:
        x_loc = float(bc['x_location'])
        u_val = float(bc['u_prescribed'])
        d = np.abs(node_coords - x_loc)
        idx = int(np.argmin(d))
        if d[idx] > tol:
            raise ValueError('Dirichlet BC x_location does not match a unique node.')
        if idx not in constrained_map:
            constrained_map[idx] = u_val
            constrained_order.append(idx)
        elif not np.isclose(constrained_map[idx], u_val, rtol=1e-12, atol=1e-12):
            raise ValueError('Conflicting Dirichlet BCs specified at the same node.')
    reaction_nodes = np.array(constrained_order, dtype=int) if constrained_order else np.array([], dtype=int)
    u = np.zeros(n_nodes, dtype=float)
    if reaction_nodes.size > 0:
        u[reaction_nodes] = np.array([constrained_map[i] for i in reaction_nodes], dtype=float)
    all_idx = np.arange(n_nodes, dtype=int)
    if reaction_nodes.size > 0:
        free_mask = np.ones(n_nodes, dtype=bool)
        free_mask[reaction_nodes] = False
        free_idx = all_idx[free_mask]
    else:
        free_idx = all_idx
    if free_idx.size > 0:
        K_FF = K[np.ix_(free_idx, free_idx)]
        F_F = F[free_idx]
        if reaction_nodes.size > 0:
            K_FC = K[np.ix_(free_idx, reaction_nodes)]
            rhs = F_F - K_FC @ u[reaction_nodes]
        else:
            rhs = F_F
        u[free_idx] = np.linalg.solve(K_FF, rhs) if K_FF.size > 0 else np.array([], dtype=float)
    reactions = np.array([], dtype=float)
    if reaction_nodes.size > 0:
        r_full = K @ u - F
        reactions = r_full[reaction_nodes]
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}