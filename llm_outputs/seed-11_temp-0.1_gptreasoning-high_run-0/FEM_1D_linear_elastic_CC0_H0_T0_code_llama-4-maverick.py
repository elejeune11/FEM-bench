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
        raise ValueError('n_gauss must be 1, 2, or 3')
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    element_nodes = np.stack((np.arange(num_elements), np.arange(1, num_elements + 1)), axis=1)
    element_lengths = node_coords[element_nodes[:, 1]] - node_coords[element_nodes[:, 0]]
    element_midpoints = (node_coords[element_nodes[:, 0]] + node_coords[element_nodes[:, 1]]) / 2
    material_properties = np.zeros((num_elements, 2))
    for (i, x_center) in enumerate(element_midpoints):
        region = next((r for r in material_regions if r['coord_min'] <= x_center <= r['coord_max']), None)
        if region is None:
            raise ValueError(f'Element {i} at x={x_center} not covered by any material region')
        material_properties[i] = [region['E'], region['A']]
    gauss_weights = {1: np.array([2.0]), 2: np.array([1.0, 1.0]), 3: np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])}
    gauss_points = {1: np.array([0.0]), 2: np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]), 3: np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])}
    weights = gauss_weights[n_gauss]
    points = gauss_points[n_gauss]
    K_global = np.zeros((num_elements + 1, num_elements + 1))
    F_global = np.zeros(num_elements + 1)
    for (i, (n1, n2)) in enumerate(element_nodes):
        (E, A) = material_properties[i]
        L = element_lengths[i]
        k_local = E * A / L * np.array([[1, -1], [-1, 1]])
        K_global[n1, n1] += k_local[0, 0]
        K_global[n1, n2] += k_local[0, 1]
        K_global[n2, n1] += k_local[1, 0]
        K_global[n2, n2] += k_local[1, 1]
        f_local = np.zeros(2)
        for (j, xi) in enumerate(points):
            x = (node_coords[n2] - node_coords[n1]) * (xi + 1) / 2 + node_coords[n1]
            N = np.array([(1 - xi) / 2, (1 + xi) / 2])
            f_local += weights[j] * body_force_fn(x) * N * L / 2
        F_global[n1] += f_local[0]
        F_global[n2] += f_local[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
            if np.abs(node_coords[node_idx] - bc['x_location']) > 1e-06:
                raise ValueError(f"Neumann BC at x={bc['x_location']} does not match any node")
            F_global[node_idx] += bc['load_mag']
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        node_idx = np.argmin(np.abs(node_coords - bc['x_location']))
        if np.abs(node_coords[node_idx] - bc['x_location']) > 1e-06:
            raise ValueError(f"Dirichlet BC at x={bc['x_location']} does not match any node")
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(bc['u_prescribed'])
    free_nodes = np.setdiff1d(np.arange(num_elements + 1), dirichlet_nodes)
    K_reduced = K_global[np.ix_(free_nodes, free_nodes)]
    F_reduced = F_global[free_nodes] - K_global[np.ix_(free_nodes, dirichlet_nodes)] @ dirichlet_values
    try:
        u_free = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError('Singular reduced stiffness matrix') from e
    displacements = np.zeros(num_elements + 1)
    displacements[free_nodes] = u_free
    displacements[dirichlet_nodes] = dirichlet_values
    reactions = K_global[dirichlet_nodes, :] @ displacements - F_global[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}