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
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if not isinstance(num_elements, int) or num_elements < 1:
        raise ValueError('num_elements must be a positive integer.')
    if n_gauss is None:
        n_gauss = 2
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be in {1, 2, 3}.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    h = node_coords[1] - node_coords[0] if n_nodes > 1 else x_max - x_min
    tol_node = max(1e-12, 1e-12 * abs(h))
    tol_region = max(1e-12, 1e-12 * abs(x_max - x_min))
    E_e = np.zeros(num_elements, dtype=float)
    A_e = np.zeros(num_elements, dtype=float)
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        x_center = 0.5 * (x1 + x2)
        matches = []
        for reg in material_regions:
            cmin = float(reg['coord_min'])
            cmax = float(reg['coord_max'])
            if x_center >= cmin - tol_region and x_center <= cmax + tol_region:
                matches.append(reg)
        if len(matches) != 1:
            raise ValueError('Each element midpoint must fall into exactly one material region.')
        E_e[e] = float(matches[0]['E'])
        A_e[e] = float(matches[0]['A'])
    if n_gauss == 1:
        gauss_xi = np.array([0.0])
        gauss_w = np.array([2.0])
    elif n_gauss == 2:
        a = 1.0 / np.sqrt(3.0)
        gauss_xi = np.array([-a, a])
        gauss_w = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        gauss_xi = np.array([-a, 0.0, a])
        gauss_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        i = e
        j = e + 1
        x1 = node_coords[i]
        x2 = node_coords[j]
        L = x2 - x1
        if L <= 0.0:
            raise ValueError('Non-positive element length encountered.')
        E = E_e[e]
        A = A_e[e]
        k_local = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        K[i, i] += k_local[0, 0]
        K[i, j] += k_local[0, 1]
        K[j, i] += k_local[1, 0]
        K[j, j] += k_local[1, 1]
        fe = np.zeros(2, dtype=float)
        J = L / 2.0
        for k in range(len(gauss_xi)):
            xi = gauss_xi[k]
            w = gauss_w[k]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            x = N1 * x1 + N2 * x2
            fval = float(body_force_fn(x))
            fe[0] += N1 * fval * J * w
            fe[1] += N2 * fval * J * w
        F[i] += fe[0]
        F[j] += fe[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = float(bc['x_location'])
            load = float(bc['load_mag'])
            diffs = np.abs(node_coords - x_loc)
            idxs = np.where(diffs <= tol_node)[0]
            if len(idxs) != 1:
                raise ValueError('Neumann BC x_location must match a unique mesh node.')
            F[idxs[0]] += load
    dir_nodes = []
    dir_vals = []
    seen = {}
    for bc in dirichlet_bc_list:
        x_loc = float(bc['x_location'])
        u_val = float(bc['u_prescribed'])
        diffs = np.abs(node_coords - x_loc)
        idxs = np.where(diffs <= tol_node)[0]
        if len(idxs) != 1:
            raise ValueError('Dirichlet BC x_location must match a unique mesh node.')
        idx = int(idxs[0])
        if idx in seen:
            prev = seen[idx]
            if not np.isclose(prev, u_val, rtol=0.0, atol=max(1e-12, 1e-12 * max(1.0, abs(prev), abs(u_val)))):
                raise ValueError('Conflicting Dirichlet BCs specified at the same node.')
        else:
            seen[idx] = u_val
            dir_nodes.append(idx)
            dir_vals.append(u_val)
    reaction_nodes = np.array(dir_nodes, dtype=int)
    n_dirichlet = len(reaction_nodes)
    u = np.zeros(n_nodes, dtype=float)
    if n_dirichlet > 0:
        u[reaction_nodes] = np.array(dir_vals, dtype=float)
    if n_dirichlet < n_nodes:
        free_mask = np.ones(n_nodes, dtype=bool)
        free_mask[reaction_nodes] = False
        free_idx = np.nonzero(free_mask)[0]
        if free_idx.size > 0:
            K_ff = K[np.ix_(free_idx, free_idx)]
            if n_dirichlet > 0:
                K_fc = K[np.ix_(free_idx, reaction_nodes)]
                rhs = F[free_idx] - K_fc @ u[reaction_nodes]
            else:
                rhs = F[free_idx]
            u_free = np.linalg.solve(K_ff, rhs)
            u[free_idx] = u_free
    else:
        pass
    residual = K @ u - F
    reactions = residual[reaction_nodes] if n_dirichlet > 0 else np.zeros(0, dtype=float)
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}