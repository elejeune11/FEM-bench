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
    if n_gauss == 1:
        gauss_points = np.array([0.0])
        gauss_weights = np.array([2.0])
    elif n_gauss == 2:
        gauss_points = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
        gauss_weights = np.array([1.0, 1.0])
    else:
        gauss_points = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        gauss_weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    E_vals = np.zeros(num_elements)
    A_vals = np.zeros(num_elements)
    for elem_idx in range(num_elements):
        x_center = (node_coords[elem_idx] + node_coords[elem_idx + 1]) / 2.0
        found = False
        for region in material_regions:
            if region['coord_min'] <= x_center <= region['coord_max']:
                if found:
                    raise ValueError(f'Element {elem_idx} (midpoint {x_center}) belongs to multiple regions')
                E_vals[elem_idx] = region['E']
                A_vals[elem_idx] = region['A']
                found = True
        if not found:
            raise ValueError(f'Element {elem_idx} (midpoint {x_center}) not covered by any material region')
    K_global = np.zeros((num_nodes, num_nodes))
    f_global = np.zeros(num_nodes)

    def local_stiffness(EA, L):
        return EA / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    for elem_idx in range(num_elements):
        node1 = elem_idx
        node2 = elem_idx + 1
        x1 = node_coords[node1]
        x2 = node_coords[node2]
        L = x2 - x1
        E = E_vals[elem_idx]
        A = A_vals[elem_idx]
        EA = E * A
        K_local = local_stiffness(EA, L)
        K_global[node1, node1] += K_local[0, 0]
        K_global[node1, node2] += K_local[0, 1]
        K_global[node2, node1] += K_local[1, 0]
        K_global[node2, node2] += K_local[1, 1]
        for gp_idx in range(n_gauss):
            xi = gauss_points[gp_idx]
            wt = gauss_weights[gp_idx]
            x_phys = x1 + (xi + 1.0) / 2.0 * L
            jacobian = L / 2.0
            N1 = (1.0 - xi) / 2.0
            N2 = (1.0 + xi) / 2.0
            f_val = body_force_fn(x_phys)
            integrand_weight = f_val * jacobian * wt
            f_global[node1] += N1 * integrand_weight
            f_global[node2] += N2 * integrand_weight
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            node_idx = None
            for (i, x_node) in enumerate(node_coords):
                if np.isclose(x_node, x_loc):
                    if node_idx is not None:
                        raise ValueError(f'Multiple nodes match Neumann BC location {x_loc}')
                    node_idx = i
            if node_idx is None:
                raise ValueError(f'No node found at Neumann BC location {x_loc}')
            f_global[node_idx] += load_mag
    dirichlet_nodes = []
    dirichlet_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_prescribed = bc['u_prescribed']
        node_idx = None
        for (i, x_node) in enumerate(node_coords):
            if np.isclose(x_node, x_loc):
                if node_idx is not None:
                    raise ValueError(f'Multiple nodes match Dirichlet BC location {x_loc}')
                node_idx = i
        if node_idx is None:
            raise ValueError(f'No node found at Dirichlet BC location {x_loc}')
        dirichlet_nodes.append(node_idx)
        dirichlet_values.append(u_prescribed)
    dirichlet_nodes = np.array(dirichlet_nodes, dtype=int)
    dirichlet_values = np.array(dirichlet_values)
    free_nodes = np.array([i for i in range(num_nodes) if i not in dirichlet_nodes], dtype=int)
    K_reduced = K_global[np.ix_(free_nodes, free_nodes)]
    f_reduced = f_global[free_nodes].copy()
    for (i, d_node) in enumerate(dirichlet_nodes):
        for (j, f_node) in enumerate(free_nodes):
            f_reduced[j] -= K_global[f_node, d_node] * dirichlet_values[i]
    u_reduced = np.linalg.solve(K_reduced, f_reduced)
    u_full = np.zeros(num_nodes)
    u_full[free_nodes] = u_reduced
    u_full[dirichlet_nodes] = dirichlet_values
    reactions = np.zeros(len(dirichlet_nodes))
    for (i, d_node) in enumerate(dirichlet_nodes):
        reactions[i] = 0.0
        for j in range(num_nodes):
            reactions[i] += K_global[d_node, j] * u_full[j]
        reactions[i] -= f_global[d_node]
    return {'displacements': u_full, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': dirichlet_nodes}