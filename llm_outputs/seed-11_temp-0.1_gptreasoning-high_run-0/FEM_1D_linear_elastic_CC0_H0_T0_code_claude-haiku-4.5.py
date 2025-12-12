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
        raise ValueError(f'n_gauss must be in {{1, 2, 3}}, got {n_gauss}')
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    gauss_rules = {1: (np.array([0.0]), np.array([2.0])), 2: (np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]), np.array([1.0, 1.0])), 3: (np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]))}
    (gauss_pts, gauss_wts) = gauss_rules[n_gauss]
    E_array = np.zeros(num_elements)
    A_array = np.zeros(num_elements)
    for e in range(num_elements):
        x_center = 0.5 * (node_coords[e] + node_coords[e + 1])
        found = False
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                if found:
                    raise ValueError(f'Element {e} (center x={x_center}) falls in multiple regions')
                E_array[e] = region['E']
                A_array[e] = region['A']
                found = True
        if not found:
            raise ValueError(f'Element {e} (center x={x_center}) not covered by any material region')
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        h_e = x2 - x1
        E = E_array[e]
        A = A_array[e]
        k_e = E * A / h_e * np.array([[1, -1], [-1, 1]])
        f_e = np.zeros(2)
        for gp in range(n_gauss):
            xi = gauss_pts[gp]
            wt = gauss_wts[gp]
            x = 0.5 * (x1 + x2) + 0.5 * h_e * xi
            f_x = body_force_fn(x)
            J = h_e / 2.0
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            f_e[0] += N1 * f_x * J * wt
            f_e[1] += N2 * f_x * J * wt
        nodes = [e, e + 1]
        for i in range(2):
            for j in range(2):
                K_global[nodes[i], nodes[j]] += k_e[i, j]
            F_global[nodes[i]] += f_e[i]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load = bc['load_mag']
            node_idx = None
            for n in range(num_nodes):
                if np.isclose(node_coords[n], x_loc):
                    if node_idx is not None:
                        raise ValueError(f'Multiple nodes at x={x_loc}')
                    node_idx = n
            if node_idx is None:
                raise ValueError(f'No node found at x={x_loc} for Neumann BC')
            F_global[node_idx] += load
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_val = bc['u_prescribed']
        node_idx = None
        for n in range(num_nodes):
            if np.isclose(node_coords[n], x_loc):
                if node_idx is not None:
                    raise ValueError(f'Multiple nodes at x={x_loc}')
                node_idx = n
        if node_idx is None:
            raise ValueError(f'No node found at x={x_loc} for Dirichlet BC')
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(u_val)
    dirichlet_nodes = np.array(dirichlet_nodes)
    dirichlet_values = np.array(dirichlet_values)
    all_nodes = np.arange(num_nodes)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes)
    K_reduced = K_global[np.ix_(free_nodes, free_nodes)]
    F_reduced = F_global[free_nodes].copy()
    for (i, d_node) in enumerate(dirichlet_nodes):
        u_d = dirichlet_values[i]
        for (j, f_node) in enumerate(free_nodes):
            F_reduced[j] -= K_global[f_node, d_node] * u_d
    u_free = np.linalg.solve(K_reduced, F_reduced)
    u_global = np.zeros(num_nodes)
    u_global[free_nodes] = u_free
    u_global[dirichlet_nodes] = dirichlet_values
    reactions = np.zeros(len(dirichlet_nodes))
    for (i, d_node) in enumerate(dirichlet_nodes):
        reactions[i] = np.dot(K_global[d_node, :], u_global)
    return {'displacements': u_global, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}