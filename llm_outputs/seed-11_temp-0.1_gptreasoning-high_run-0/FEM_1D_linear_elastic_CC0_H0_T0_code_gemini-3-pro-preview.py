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
    if n_gauss not in [1, 2, 3]:
        raise ValueError(f'n_gauss must be 1, 2, or 3. Got {n_gauss}.')
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    num_nodes = len(node_coords)
    element_length = (x_max - x_min) / num_elements
    if n_gauss == 1:
        xi_list = [0.0]
        w_list = [2.0]
    elif n_gauss == 2:
        val = 1.0 / np.sqrt(3.0)
        xi_list = [-val, val]
        w_list = [1.0, 1.0]
    else:
        val = np.sqrt(3.0 / 5.0)
        xi_list = [0.0, -val, val]
        w_list = [8.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0]
    K = np.zeros((num_nodes, num_nodes))
    F = np.zeros(num_nodes)
    for e in range(num_elements):
        (n1, n2) = (e, e + 1)
        (x1, x2) = (node_coords[n1], node_coords[n2])
        x_center = (x1 + x2) / 2.0
        matched_region = None
        count = 0
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                matched_region = region
                count += 1
        if count != 1:
            raise ValueError(f'Element {e} centered at {x_center} is covered by {count} material regions.')
        E = matched_region['E']
        A = matched_region['A']
        k_val = E * A / element_length
        k_elem = np.array([[k_val, -k_val], [-k_val, k_val]])
        K[n1, n1] += k_elem[0, 0]
        K[n1, n2] += k_elem[0, 1]
        K[n2, n1] += k_elem[1, 0]
        K[n2, n2] += k_elem[1, 1]
        f_elem = np.zeros(2)
        det_J = element_length / 2.0
        for (xi, w) in zip(xi_list, w_list):
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            x_g = x_center + xi * det_J
            f_val = body_force_fn(x_g)
            contrib = f_val * det_J * w
            f_elem[0] += N1 * contrib
            f_elem[1] += N2 * contrib
        F[n1] += f_elem[0]
        F[n2] += f_elem[1]
    tol = 1e-09 * (x_max - x_min)
    if neumann_bc_list is not None:
        for nbc in neumann_bc_list:
            loc = nbc['x_location']
            mag = nbc['load_mag']
            dists = np.abs(node_coords - loc)
            node_idx = np.argmin(dists)
            if dists[node_idx] > tol:
                raise ValueError(f'Neumann BC at {loc} does not match any mesh node.')
            F[node_idx] += mag
    prescribed_dofs = []
    prescribed_vals = []
    bc_node_map = {}
    ordered_bc_indices = []
    for dbc in dirichlet_bc_list:
        loc = dbc['x_location']
        val = dbc['u_prescribed']
        dists = np.abs(node_coords - loc)
        node_idx = int(np.argmin(dists))
        if dists[node_idx] > tol:
            raise ValueError(f'Dirichlet BC at {loc} does not match any mesh node.')
        if node_idx in bc_node_map:
            if not np.isclose(bc_node_map[node_idx], val):
                raise ValueError(f'Conflicting Dirichlet BCs at node {node_idx}.')
        else:
            bc_node_map[node_idx] = val
            prescribed_dofs.append(node_idx)
            prescribed_vals.append(val)
        ordered_bc_indices.append(node_idx)
    prescribed_dofs = np.array(prescribed_dofs, dtype=int)
    prescribed_vals = np.array(prescribed_vals, dtype=float)
    all_dofs = np.arange(num_nodes)
    free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)
    u = np.zeros(num_nodes)
    u[prescribed_dofs] = prescribed_vals
    if len(free_dofs) > 0:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        K_fp = K[np.ix_(free_dofs, prescribed_dofs)]
        F_f = F[free_dofs]
        rhs = F_f - K_fp @ prescribed_vals
        try:
            u_f = np.linalg.solve(K_ff, rhs)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError('Stiffness matrix is singular (insufficient BCs?).')
        u[free_dofs] = u_f
    F_int = K @ u
    R_total = F_int - F
    reactions_out = []
    reaction_nodes_out = []
    for node_idx in ordered_bc_indices:
        reactions_out.append(R_total[node_idx])
        reaction_nodes_out.append(node_idx)
    return {'displacements': u, 'reactions': np.array(reactions_out), 'node_coords': node_coords, 'reaction_nodes': np.array(reaction_nodes_out)}