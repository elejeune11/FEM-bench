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
    import numpy as np
    if num_elements <= 0:
        raise ValueError('num_elements must be a positive integer.')
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    dx = (x_max - x_min) / num_elements
    tol = 1e-10
    if n_gauss == 1:
        xi_gp = np.array([0.0])
        w_gp = np.array([2.0])
    elif n_gauss == 2:
        xi_gp = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w_gp = np.array([1.0, 1.0])
    else:
        xi_gp = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        w_gp = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def x_to_node_index(x_value: float) -> int:
        idx_float = (x_value - x_min) / dx
        idx = int(np.rint(idx_float))
        if idx < 0 or idx >= n_nodes:
            raise ValueError(f'x_location {x_value} is outside the domain.')
        if abs(node_coords[idx] - x_value) > max(tol, 1e-08 * max(1.0, abs(x_value))):
            raise ValueError(f'x_location {x_value} does not coincide with a mesh node.')
        return idx

    def element_EA(x_mid: float) -> (float, float):
        for reg in material_regions:
            cmin = reg.get('coord_min', None)
            cmax = reg.get('coord_max', None)
            E = reg.get('E', None)
            A = reg.get('A', None)
            if cmin is None or cmax is None or E is None or (A is None):
                raise ValueError("Each material region must have 'coord_min', 'coord_max', 'E', and 'A'.")
            if x_mid >= cmin - tol and x_mid <= cmax + tol:
                if E <= 0.0 or A <= 0.0:
                    raise ValueError("Material properties 'E' and 'A' must be positive.")
                return (float(E), float(A))
        raise ValueError(f'No material region found covering element at x={x_mid}.')
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        L = x2 - x1
        if L <= 0.0:
            raise ValueError('Non-positive element length encountered.')
        x_mid = 0.5 * (x1 + x2)
        (E, A) = element_EA(x_mid)
        k_local = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        f_local = np.zeros(2, dtype=float)
        J = L / 2.0
        for (xi, w) in zip(xi_gp, w_gp):
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            x_gp = x1 * N1 + x2 * N2
            qx = float(body_force_fn(x_gp))
            f_local[0] += N1 * qx * J * w
            f_local[1] += N2 * qx * J * w
        dofs = [e, e + 1]
        K[np.ix_(dofs, dofs)] += k_local
        F[dofs[0]] += f_local[0]
        F[dofs[1]] += f_local[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc.get('x_location', None)
            load_mag = bc.get('load_mag', None)
            if x_loc is None or load_mag is None:
                raise ValueError("Each Neumann BC must have 'x_location' and 'load_mag'.")
            idx = x_to_node_index(float(x_loc))
            F[idx] += float(load_mag)
    if len(dirichlet_bc_list) == 0:
        raise ValueError('At least one Dirichlet boundary condition is required.')
    bc_nodes_ordered = []
    bc_u_ordered = []
    for bc in dirichlet_bc_list:
        x_loc = bc.get('x_location', None)
        u_pres = bc.get('u_prescribed', None)
        if x_loc is None or u_pres is None:
            raise ValueError("Each Dirichlet BC must have 'x_location' and 'u_prescribed'.")
        idx = x_to_node_index(float(x_loc))
        bc_nodes_ordered.append(idx)
        bc_u_ordered.append(float(u_pres))
    visited = {}
    unique_nodes = []
    unique_u = []
    for (idx, uval) in zip(bc_nodes_ordered, bc_u_ordered):
        if idx in visited:
            if abs(visited[idx] - uval) > 1e-12:
                raise ValueError(f'Conflicting Dirichlet BCs at node {idx}.')
        else:
            visited[idx] = uval
            unique_nodes.append(idx)
            unique_u.append(uval)
    unique_nodes = np.array(unique_nodes, dtype=int)
    unique_u = np.array(unique_u, dtype=float)
    all_nodes = np.arange(n_nodes, dtype=int)
    free_mask = np.ones(n_nodes, dtype=bool)
    free_mask[unique_nodes] = False
    free_nodes = all_nodes[free_mask]
    u = np.zeros(n_nodes, dtype=float)
    u[unique_nodes] = unique_u
    if free_nodes.size > 0:
        K_ff = K[np.ix_(free_nodes, free_nodes)]
        K_fc = K[np.ix_(free_nodes, unique_nodes)]
        F_f = F[free_nodes]
        rhs = F_f - K_fc @ unique_u
        u_free = np.linalg.solve(K_ff, rhs)
        u[free_nodes] = u_free
    else:
        pass
    R_unique = K[np.ix_(unique_nodes, all_nodes)] @ u - F[unique_nodes]
    reaction_map = {node: R_val for (node, R_val) in zip(unique_nodes.tolist(), R_unique.tolist())}
    reactions_ordered = np.array([reaction_map[idx] for idx in bc_nodes_ordered], dtype=float)
    reaction_nodes = np.array(bc_nodes_ordered, dtype=int)
    return {'displacements': u, 'reactions': reactions_ordered, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}