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
    if not isinstance(num_elements, int) or num_elements < 1:
        raise ValueError('num_elements must be a positive integer.')
    if n_gauss is None:
        n_gauss = 2
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be in {1, 2, 3}.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    Ltot = x_max - x_min
    tol = 1e-12 * max(1.0, abs(Ltot))
    if n_gauss == 1:
        gauss_xi = np.array([0.0])
        gauss_w = np.array([2.0])
    elif n_gauss == 2:
        gauss_xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        gauss_w = np.array([1.0, 1.0])
    else:
        gauss_xi = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    for e in range(num_elements):
        n1 = e
        n2 = e + 1
        x1 = node_coords[n1]
        x2 = node_coords[n2]
        Le = x2 - x1
        if Le <= 0.0:
            raise ValueError(f'Non-positive element length encountered at element {e}.')
        xc = 0.5 * (x1 + x2)
        matches = 0
        E_val = None
        A_val = None
        for region in material_regions:
            if not all((k in region for k in ('coord_min', 'coord_max', 'E', 'A'))):
                raise ValueError("Each material region must contain keys: 'coord_min', 'coord_max', 'E', 'A'.")
            rmin = float(region['coord_min'])
            rmax = float(region['coord_max'])
            if xc >= rmin - tol and xc <= rmax + tol:
                matches += 1
                E_val = float(region['E'])
                A_val = float(region['A'])
        if matches != 1:
            raise ValueError(f'Element {e} with midpoint {xc} falls into {matches} material regions; expected exactly 1.')
        kfac = E_val * A_val / Le
        K[n1, n1] += kfac
        K[n1, n2] -= kfac
        K[n2, n1] -= kfac
        K[n2, n2] += kfac
        fe1 = 0.0
        fe2 = 0.0
        J = Le / 2.0
        for gi in range(gauss_xi.size):
            xi_g = gauss_xi[gi]
            w_g = gauss_w[gi]
            N1 = 0.5 * (1.0 - xi_g)
            N2 = 0.5 * (1.0 + xi_g)
            xg = N1 * x1 + N2 * x2
            fx = float(body_force_fn(xg))
            fe1 += N1 * fx * J * w_g
            fe2 += N2 * fx * J * w_g
        F[n1] += fe1
        F[n2] += fe2
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            if not all((k in bc for k in ('x_location', 'load_mag'))):
                raise ValueError("Each Neumann BC must contain keys: 'x_location', 'load_mag'.")
            x_loc = float(bc['x_location'])
            load = float(bc['load_mag'])
            diffs = np.abs(node_coords - x_loc)
            idxs = np.nonzero(diffs <= tol)[0]
            if idxs.size != 1:
                raise ValueError(f'Neumann BC at x={x_loc} does not match a unique node.')
            F[int(idxs[0])] += load
    constrained_nodes = []
    constrained_vals = []
    if dirichlet_bc_list is not None and len(dirichlet_bc_list) > 0:
        seen = {}
        for bc in dirichlet_bc_list:
            if not all((k in bc for k in ('x_location', 'u_prescribed'))):
                raise ValueError("Each Dirichlet BC must contain keys: 'x_location', 'u_prescribed'.")
            x_loc = float(bc['x_location'])
            u_val = float(bc['u_prescribed'])
            diffs = np.abs(node_coords - x_loc)
            idxs = np.nonzero(diffs <= tol)[0]
            if idxs.size != 1:
                raise ValueError(f'Dirichlet BC at x={x_loc} does not match a unique node.')
            node = int(idxs[0])
            if node in seen:
                j = seen[node]
                if not np.isclose(constrained_vals[j], u_val, rtol=0.0, atol=max(tol, 1e-12 * (1.0 + abs(u_val)))):
                    raise ValueError(f'Conflicting Dirichlet BCs at node {node}.')
            else:
                seen[node] = len(constrained_nodes)
                constrained_nodes.append(node)
                constrained_vals.append(u_val)
    constrained_nodes_arr = np.array(constrained_nodes, dtype=int)
    constrained_vals_arr = np.array(constrained_vals, dtype=float)
    u = np.zeros(n_nodes, dtype=float)
    if constrained_nodes_arr.size > 0:
        u[constrained_nodes_arr] = constrained_vals_arr
    all_idx = np.arange(n_nodes, dtype=int)
    if constrained_nodes_arr.size > 0:
        mask = np.ones(n_nodes, dtype=bool)
        mask[constrained_nodes_arr] = False
        free_idx = all_idx[mask]
    else:
        free_idx = all_idx
    if free_idx.size > 0:
        K_ff = K[np.ix_(free_idx, free_idx)]
        F_f = F[free_idx]
        if constrained_nodes_arr.size > 0:
            K_fc = K[np.ix_(free_idx, constrained_nodes_arr)]
            rhs = F_f - K_fc @ constrained_vals_arr
        else:
            rhs = F_f
        u_free = np.linalg.solve(K_ff, rhs)
        u[free_idx] = u_free
    R_full = K @ u - F
    reactions = R_full[constrained_nodes_arr] if constrained_nodes_arr.size > 0 else np.array([], dtype=float)
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': constrained_nodes_arr}