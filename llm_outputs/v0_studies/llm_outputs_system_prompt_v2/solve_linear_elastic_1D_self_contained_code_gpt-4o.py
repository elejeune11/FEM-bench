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
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        gauss_weights = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError('n_gauss must be 1, 2, or 3.')
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    element_length = (x_max - x_min) / num_elements
    K_global = np.zeros((num_elements + 1, num_elements + 1))
    F_global = np.zeros(num_elements + 1)
    for i in range(num_elements):
        x1 = node_coords[i]
        x2 = node_coords[i + 1]
        element_stiffness = np.zeros((2, 2))
        element_force = np.zeros(2)
        for region in material_regions:
            if region['coord_min'] <= x1 and region['coord_max'] >= x2:
                E = region['E']
                A = region['A']
                break
        else:
            raise ValueError('Element does not fit within any material region.')
        for (gp, gw) in zip(gauss_points, gauss_weights):
            xi = (x2 - x1) / 2 * gp + (x1 + x2) / 2
            N = np.array([0.5 * (1 - gp), 0.5 * (1 + gp)])
            B = np.array([-0.5 / element_length, 0.5 / element_length])
            element_stiffness += gw * E * A * np.outer(B, B) * element_length / 2
            element_force += gw * body_force_fn(xi) * N * element_length / 2
        K_global[i:i + 2, i:i + 2] += element_stiffness
        F_global[i:i + 2] += element_force
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_index = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_index] += load_mag
    dirichlet_indices = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_index = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_indices.append(node_index)
        dirichlet_values.append(u_prescribed)
    free_indices = np.setdiff1d(np.arange(num_elements + 1), dirichlet_indices)
    K_ff = K_global[np.ix_(free_indices, free_indices)]
    F_f = F_global[free_indices] - K_global[np.ix_(free_indices, dirichlet_indices)] @ np.array(dirichlet_values)
    displacements = np.zeros(num_elements + 1)
    displacements[free_indices] = np.linalg.solve(K_ff, F_f)
    displacements[dirichlet_indices] = dirichlet_values
    reactions = K_global[dirichlet_indices, :] @ displacements - F_global[dirichlet_indices]
    return {'displacements': displacements, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': np.array(dirichlet_indices)}