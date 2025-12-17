def FEM_1D_linear_elastic_CC0_H0_T0(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    import numpy as np
    from typing import Callable, List, Dict, Optional
    import pytest
    if not x_max > x_min:
        raise ValueError('x_max must be greater than x_min.')
    if not (isinstance(num_elements, int) and num_elements >= 1):
        raise ValueError('num_elements must be a positive integer.')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be in {1, 2, 3}.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    L = x_max - x_min
    tol = 1e-12 * max(1.0, abs(L))
    E_e = np.empty(num_elements, dtype=float)
    A_e = np.empty(num_elements, dtype=float)
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        x_center = 0.5 * (x1 + x2)
        matches = []
        for reg in material_regions:
            cmin = reg['coord_min']
            cmax = reg['coord_max']
            if x_center >= cmin - tol and x_center <= cmax + tol:
                matches.append(reg)
        if len(matches) != 1:
            raise ValueError(f'Element {e} with center {x_center} not covered by exactly one material region.')
        E_e[e] = matches[0]['E']
        A_e[e] = matches[0]['A']
    if n_gauss == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif n_gauss == 2:
        inv_sqrt3 = 1.0 / np.sqrt(3.0)
        gauss_pts = np.array([-inv_sqrt3, inv_sqrt3])
        gauss_wts = np.array([1.0, 1.0])
    else:
        r = np.sqrt(3.0 / 5.0)
        gauss_pts = np.array([-r, 0.0, r])
        gauss_wts = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        i = e
        j = e + 1
        x1 = node_coords[i]
        x2 = node_coords[j]
        he = x2 - x1
        if he <= 0.0:
            raise ValueError('Non-positive element length encountered.')
        Ee = E_e[e]
        Ae = A_e[e]
        coef = Ee * Ae / he
        ke = coef * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        K[i, i] += ke[0, 0]
        K[i, j] += ke[0, 1]
        K[j, i] += ke[1, 0]
        K[j, j] += ke[1, 1]
        fe = np.zeros(2, dtype=float)
        J = he / 2.0
        for gp, w in zip(gauss_pts, gauss_wts):
            N1 = 0.5 * (1.0 - gp)
            N2 = 0.5 * (1.0 + gp)
            xg = N1 * x1 + N2 * x2
            fg = body_force_fn(xg)
            fe[0] += N1 * fg * J * w
            fe[1] += N2 * fg * J * w
        F[i] += fe[0]
        F[j] += fe[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc['x_location']
            load_mag = bc['load_mag']
            idxs = np.where(np.abs(node_coords - x_loc) <= tol)[0]
            if idxs.size != 1:
                raise ValueError(f'Neumann BC x_location={x_loc} does not match a unique node.')
            F[idxs[0]] += load_mag
    fixed_nodes_order = []
    fixed_values_order = []
    fixed_map = {}
    for bc in dirichlet_bc_list:
        x_loc = bc['x_location']
        u_val = bc['u_prescribed']
        idxs = np.where(np.abs(node_coords - x_loc) <= tol)[0]
        if idxs.size != 1:
            raise ValueError(f'Dirichlet BC x_location={x_loc} does not match a unique node.')
        idx = int(idxs[0])
        if idx in fixed_map:
            if not np.isclose(fixed_map[idx], u_val, rtol=0.0, atol=tol):
                raise ValueError(f'Conflicting Dirichlet BCs at node {idx} (x={node_coords[idx]}).')
        else:
            fixed_map[idx] = float(u_val)
            fixed_nodes_order.append(idx)
            fixed_values_order.append(float(u_val))
    reaction_nodes = np.array(fixed_nodes_order, dtype=int)
    u_prescribed = np.array(fixed_values_order, dtype=float)
    u = np.zeros(n_nodes, dtype=float)
    if reaction_nodes.size > 0:
        u[reaction_nodes] = u_prescribed
    if reaction_nodes.size < n_nodes:
        mask_free = np.ones(n_nodes, dtype=bool)
        mask_free[reaction_nodes] = False
        free_nodes = np.nonzero(mask_free)[0]
        K_ff = K[np.ix_(free_nodes, free_nodes)]
        K_fc = K[np.ix_(free_nodes, reaction_nodes)] if reaction_nodes.size > 0 else np.zeros((free_nodes.size, 0), dtype=float)
        rhs = F[free_nodes] - (K_fc @ u[reaction_nodes] if reaction_nodes.size > 0 else 0.0)
        u_free = np.linalg.solve(K_ff, rhs)
        u[free_nodes] = u_free
    else:
        free_nodes = np.array([], dtype=int)
    residual = K @ u - F
    reactions = residual[reaction_nodes] if reaction_nodes.size > 0 else np.array([], dtype=float)
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}