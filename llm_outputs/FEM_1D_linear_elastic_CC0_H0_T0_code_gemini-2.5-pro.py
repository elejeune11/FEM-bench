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
    if n_gauss not in {1, 2, 3}:
        raise ValueError('n_gauss must be 1, 2, or 3.')
    num_nodes = num_elements + 1
    element_length = (x_max - x_min) / num_elements
    node_coords = np.linspace(x_min, x_max, num_nodes)
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        val = 1.0 / np.sqrt(3.0)
        gauss_points = np.array([-val, val])
        gauss_weights = np.array([1.0, 1.0])
    else:
        val = np.sqrt(3.0 / 5.0)
        gauss_points = np.array([-val, 0.0, val])
        gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
    for e_idx in range(num_elements):
        (node1_idx, node2_idx) = (e_idx, e_idx + 1)
        (x1, x2) = (node_coords[node1_idx], node_coords[node2_idx])
        x_center = (x1 + x2) / 2.0
        matching_regions = []
        for region in material_regions:
            is_in_region = region['coord_min'] <= x_center < region['coord_max'] or (np.isclose(x_center, region['coord_max']) and np.isclose(x_center, x_max))
            if is_in_region:
                matching_regions.append(region)
        if len(matching_regions) != 1:
            raise ValueError(f'Element {e_idx} midpoint {x_center} is covered by {len(matching_regions)} material regions, expected 1.')
        props = matching_regions[0]
        (E, A) = (props['E'], props['A'])
        k_element = E * A / element_length * np.array([[1, -1], [-1, 1]])
        f_element = np.zeros(2)
        jacobian = element_length / 2.0
        for (gp, gw) in zip(gauss_points, gauss_weights):
            N1 = (1.0 - gp) / 2.0
            N2 = (1.0 + gp) / 2.0
            N_vec = np.array([N1, N2])
            x_gp = x1 * N1 + x2 * N2
            f_val = body_force_fn(x_gp)
            f_element += N_vec * f_val * jacobian * gw
        indices = np.array([node1_idx, node2_idx])
        K_global[np.ix_(indices, indices)] += k_element
        F_global[indices] += f_element
    if neumann_bc_list:
        for bc in neumann_bc_list:
            (x_loc, load) = (bc['x_location'], bc['load_mag'])
            matches = np.where(np.isclose(node_coords, x_loc))[0]
            if len(matches) != 1:
                raise ValueError(f'Neumann BC at x={x_loc} does not match a unique node.')
            F_global[matches[0]] += load
    dirichlet_nodes_list = []
    dirichlet_values_list = []
    seen_nodes = set()
    for bc in dirichlet_bc_list:
        (x_loc, u_val) = (bc['x_location'], bc['u_prescribed'])
        matches = np.where(np.isclose(node_coords, x_loc))[0]
        if len(matches) != 1:
            raise ValueError(f'Dirichlet BC at x={x_loc} does not match a unique node.')
        node_idx = matches[0]
        if node_idx in seen_nodes:
            raise ValueError(f'Multiple Dirichlet BCs specified for node {node_idx} at x={x_loc}.')
        seen_nodes.add(node_idx)
        dirichlet_nodes_list.append(node_idx)
        dirichlet_values_list.append(u_val)
    dirichlet_nodes = np.array(dirichlet_nodes_list, dtype=int)
    dirichlet_values = np.array(dirichlet_values_list)
    all_nodes = np.arange(num_nodes)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes, assume_unique=True)
    K_ff = K_global[np.ix_(free_nodes, free_nodes)]
    K_fc = K_global[np.ix_(free_nodes, dirichlet_nodes)]
    F_f = F_global[free_nodes]
    u_c = dirichlet_values
    F_reduced = F_f - K_fc @ u_c
    u_f = np.linalg.solve(K_ff, F_reduced)
    displacements = np.zeros(num_nodes)
    displacements[free_nodes] = u_f
    displacements[dirichlet_nodes] = u_c
    full_force_vector = K_global @ displacements
    reactions = full_force_vector[dirichlet_nodes] - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}