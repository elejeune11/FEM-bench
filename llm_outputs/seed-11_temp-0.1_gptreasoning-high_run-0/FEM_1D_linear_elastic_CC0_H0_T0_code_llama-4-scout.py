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
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    element_materials = []
    for i in range(num_elements):
        x_center = (node_coords[i] + node_coords[i + 1]) / 2
        for region in material_regions:
            if region['coord_min'] <= x_center < region['coord_max']:
                element_materials.append({'E': region['E'], 'A': region['A']})
                break
    if len(element_materials) != num_elements:
        raise ValueError('Every element must fall into exactly one material region.')
    K_global = np.zeros((num_elements + 1, num_elements + 1))
    F_global = np.zeros(num_elements + 1)
    for i in range(num_elements):
        (node_i, node_j) = (i, i + 1)
        L_e = node_coords[i + 1] - node_coords[i]
        E = element_materials[i]['E']
        A = element_materials[i]['A']
        K_local = E * A / L_e * np.array([[1, -1], [-1, 1]])
        K_global[node_i:node_j + 1, node_i:node_j + 1] += K_local
        if n_gauss == 1:
            gauss_point = (node_coords[i] + node_coords[i + 1]) / 2
            F_local = body_force_fn(gauss_point) * L_e * np.array([0.5, 0.5])
        elif n_gauss == 2:
            gauss_points = [(node_coords[i] + node_coords[i + 1]) / 2 + L_e * (-1 / np.sqrt(3)) / 2, (node_coords[i] + node_coords[i + 1]) / 2 + L_e * (1 / np.sqrt(3)) / 2]
            F_local = L_e / 2 * np.array([body_force_fn(gauss_points[0]), body_force_fn(gauss_points[1])])
        elif n_gauss == 3:
            gauss_points = [(node_coords[i] + node_coords[i + 1]) / 2 + L_e * -np.sqrt(3 / 5) / 2, (node_coords[i] + node_coords[i + 1]) / 2, (node_coords[i] + node_coords[i + 1]) / 2 + L_e * np.sqrt(3 / 5) / 2]
            weights = [5 / 9, 8 / 9, 5 / 9]
            F_local = L_e / 2 * np.array([weights[0] * body_force_fn(gauss_points[0]), weights[1] * body_force_fn(gauss_points[1]), weights[2] * body_force_fn(gauss_points[2])])
        else:
            raise ValueError('n_gauss must be 1, 2, or 3.')
        F_global[node_i:node_j + 1] += F_local
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
    K_reduced = np.delete(np.delete(K_global, dirichlet_nodes, axis=0), dirichlet_nodes, axis=1)
    F_reduced = F_global.copy()
    for (i, value) in zip(dirichlet_nodes, dirichlet_values):
        F_reduced -= K_global[:, i] * value
    try:
        displacements = np.zeros(num_elements + 1)
        displacements[dirichlet_nodes] = dirichlet_values
        displacements[np.setdiff1d(range(num_elements + 1), dirichlet_nodes)] = np.linalg.solve(K_reduced, F_reduced[np.setdiff1d(range(num_elements + 1), dirichlet_nodes)])
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError('The reduced stiffness matrix is singular.')
    reactions = K_global @ displacements - F_global
    output = {'displacements': displacements, 'reactions': reactions[dirichlet_nodes], 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_nodes)}
    return output