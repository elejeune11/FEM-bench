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
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    n_nodes = len(node_coords)
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_points = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    for elem_idx in range(num_elements):
        (node1, node2) = elements[elem_idx]
        (x1, x2) = (node_coords[node1], node_coords[node2])
        L = x2 - x1
        elem_center = (x1 + x2) / 2
        (E, A) = (None, None)
        for region in material_regions:
            if region['coord_min'] <= elem_center <= region['coord_max']:
                E = region['E']
                A = region['A']
                break
        k_elem = E * A / L * np.array([[1, -1], [-1, 1]])
        f_elem = np.zeros(2)
        for gp_idx in range(n_gauss):
            xi = gauss_points[gp_idx]
            w = gauss_weights[gp_idx]
            x_phys = x1 + (xi + 1) * L / 2
            N1 = (1 - xi) / 2
            N2 = (1 + xi) / 2
            f_body = body_force_fn(x_phys)
            f_elem[0] += N1 * f_body * A * (L / 2) * w
            f_elem[1] += N2 * f_body * A * (L / 2) * w
        dofs = [node1, node2]
        for i in range(2):
            for j in range(2):
                K_global[dofs[i], dofs[j]] += k_elem[i, j]
            F_global[dofs[i]] += f_elem[i]
    if neumann_bc_list:
        for neumann_bc in neumann_bc_list:
            x_loc = neumann_bc['x_location']
            load_mag = neumann_bc['load_mag']
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            F_global[node_idx] += load_mag
    dirichlet_nodes = []
    dirichlet_values = []
    for dirichlet_bc in dirichlet_bc_list:
        x_loc = dirichlet_bc['x_location']
        u_prescribed = dirichlet_bc['u_prescribed']
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(u_prescribed)
    dirichlet_nodes = np.array(dirichlet_nodes)
    dirichlet_values = np.array(dirichlet_values)
    K_original = K_global.copy()
    F_original = F_global.copy()
    penalty = 1000000000000.0
    for (i, node_idx) in enumerate(dirichlet_nodes):
        K_global[node_idx, node_idx] += penalty
        F_global[node_idx] += penalty * dirichlet_values[i]
    displacements = np.linalg.solve(K_global, F_global)
    reactions = K_original @ displacements - F_original
    reaction_forces = reactions[dirichlet_nodes]
    return {'displacements': displacements, 'reactions': reaction_forces, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}