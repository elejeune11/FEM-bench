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
    from typing import Callable, List, Dict, Optional
    import pytest
    if num_elements <= 0:
        raise ValueError('num_elements must be positive.')
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    h = (x_max - x_min) / num_elements
    tol = max(1e-12, 1e-09 * (x_max - x_min))

    def get_material_at(x: float):
        for reg in material_regions:
            cmin = reg['coord_min']
            cmax = reg['coord_max']
            if x >= cmin - tol and x <= cmax + tol:
                return (float(reg['E']), float(reg['A']))
        raise ValueError(f'No material region found covering x={x}.')
    if n_gauss == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif n_gauss == 2:
        a = 1.0 / np.sqrt(3.0)
        gauss_pts = np.array([-a, a])
        gauss_wts = np.array([1.0, 1.0])
    else:
        a = np.sqrt(3.0 / 5.0)
        gauss_pts = np.array([-a, 0.0, a])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        n1 = e
        n2 = e + 1
        x1 = node_coords[n1]
        x2 = node_coords[n2]
        L = x2 - x1
        J = L / 2.0
        B = np.array([-1.0 / L, 1.0 / L], dtype=float)
        Ke = np.zeros((2, 2), dtype=float)
        Fe = np.zeros(2, dtype=float)
        for (xi, w) in zip(gauss_pts, gauss_wts):
            xg = 0.5 * (x1 + x2) + 0.5 * L * xi
            (E, A) = get_material_at(xg)
            EA = E * A
            N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0], dtype=float)
            b = float(body_force_fn(xg))
            Ke += w * J * EA * np.outer(B, B)
            Fe += w * J * A * b * N
        dofs = [n1, n2]
        for (i_local, I) in enumerate(dofs):
            F[I] += Fe[i_local]
            for (j_local, Jd) in enumerate(dofs):
                K[I, Jd] += Ke[i_local, j_local]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = float(bc['x_location'])
            load = float(bc['load_mag'])
            idx_candidates = np.where(np.isclose(node_coords, x_loc, atol=tol, rtol=0.0))[0]
            if idx_candidates.size == 0:
                idx_near = int(round((x_loc - x_min) / h))
                if idx_near < 0 or idx_near >= n_nodes or abs(node_coords[idx_near] - x_loc) > tol:
                    raise ValueError(f'Neumann BC location {x_loc} does not match a node.')
                idx = idx_near
            else:
                idx = int(idx_candidates[0])
            F[idx] += load
    bc_indices_order = []
    bc_values_order = []
    for bc in dirichlet_bc_list:
        x_loc = float(bc['x_location'])
        u_val = float(bc['u_prescribed'])
        idx_candidates = np.where(np.isclose(node_coords, x_loc, atol=tol, rtol=0.0))[0]
        if idx_candidates.size == 0:
            idx_near = int(round((x_loc - x_min) / h))
            if idx_near < 0 or idx_near >= n_nodes or abs(node_coords[idx_near] - x_loc) > tol:
                raise ValueError(f'Dirichlet BC location {x_loc} does not match a node.')
            idx = idx_near
        else:
            idx = int(idx_candidates[0])
        bc_indices_order.append(idx)
        bc_values_order.append(u_val)
    unique_constrained = []
    prescribed_map = {}
    for (idx, val) in zip(bc_indices_order, bc_values_order):
        prescribed_map[idx] = val
        if idx not in unique_constrained:
            unique_constrained.append(idx)
    unique_constrained = np.array(unique_constrained, dtype=int)
    u_c = np.array([prescribed_map[i] for i in unique_constrained], dtype=float)
    all_indices = np.arange(n_nodes, dtype=int)
    mask = np.ones(n_nodes, dtype=bool)
    mask[unique_constrained] = False
    free = all_indices[mask]
    u = np.zeros(n_nodes, dtype=float)
    if unique_constrained.size > 0:
        u[unique_constrained] = u_c
    if free.size > 0:
        K_ff = K[np.ix_(free, free)]
        K_fc = K[np.ix_(free, unique_constrained)]
        rhs = F[free] - K_fc @ u_c
        u_f = np.linalg.solve(K_ff, rhs)
        u[free] = u_f
    residual = K @ u - F
    reactions_order = np.array([residual[i] for i in bc_indices_order], dtype=float)
    reaction_nodes_order = np.array(bc_indices_order, dtype=int)
    return {'displacements': u, 'reactions': reactions_order, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes_order}