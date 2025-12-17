def FEM_1D_linear_elastic_CC0_H0_T0(x_min: float, x_max: float, num_elements: int, material_regions: List[Dict[str, float]], body_force_fn: Callable[[float], float], dirichlet_bc_list: List[Dict[str, float]], neumann_bc_list: Optional[List[Dict[str, float]]], n_gauss: int) -> Dict[str, np.ndarray]:
    import numpy as np
    from typing import Callable, List, Dict, Optional
    import pytest
    if not x_max > x_min:
        raise ValueError('x_max must be greater than x_min.')
    if not (isinstance(num_elements, int) and num_elements >= 1):
        raise ValueError('num_elements must be an integer >= 1.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes, dtype=float)
    if n_gauss is None:
        n_gauss = 2
    if n_gauss == 1:
        gauss_xi = np.array([0.0], dtype=float)
        gauss_w = np.array([2.0], dtype=float)
    elif n_gauss == 2:
        a = 1.0 / np.sqrt(3.0)
        gauss_xi = np.array([-a, a], dtype=float)
        gauss_w = np.array([1.0, 1.0], dtype=float)
    elif n_gauss == 3:
        a = np.sqrt(3.0 / 5.0)
        gauss_xi = np.array([-a, 0.0, a], dtype=float)
        gauss_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    else:
        raise ValueError('n_gauss must be in {1, 2, 3}.')
    if material_regions is None or len(material_regions) == 0:
        raise ValueError('At least one material region must be provided.')
    try:
        sorted_regions = sorted(material_regions, key=lambda r: float(r['coord_min']))
    except Exception:
        raise ValueError("Each material region must include 'coord_min'.")
    tol = 1e-12 * max(1.0, abs(x_max - x_min))
    E_e = np.zeros(num_elements, dtype=float)
    A_e = np.zeros(num_elements, dtype=float)
    for e in range(num_elements):
        x1 = node_coords[e]
        x2 = node_coords[e + 1]
        xc = 0.5 * (x1 + x2)
        match_count = 0
        chosen_E = None
        chosen_A = None
        for i, reg in enumerate(sorted_regions):
            if not all((k in reg for k in ('coord_min', 'coord_max', 'E', 'A'))):
                raise ValueError("Each material region must provide 'coord_min', 'coord_max', 'E', and 'A'.")
            mn = float(reg['coord_min'])
            mx = float(reg['coord_max'])
            last = i == len(sorted_regions) - 1
            inside = xc >= mn - tol and (xc < mx - tol / 2.0 if not last else xc <= mx + tol)
            if inside:
                chosen_E = float(reg['E'])
                chosen_A = float(reg['A'])
                match_count += 1
        if match_count == 0:
            raise ValueError(f'Element at x_center={xc} not covered by any material region.')
        if match_count > 1:
            raise ValueError(f'Element at x_center={xc} falls into multiple material regions.')
        E_e[e] = chosen_E
        A_e[e] = chosen_A
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        i = e
        j = e + 1
        x1 = node_coords[i]
        x2 = node_coords[j]
        le = x2 - x1
        if not le > 0:
            raise ValueError('Non-positive element length encountered.')
        Ee = E_e[e]
        Ae = A_e[e]
        ke = Ee * Ae / le * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        K[i, i] += ke[0, 0]
        K[i, j] += ke[0, 1]
        K[j, i] += ke[1, 0]
        K[j, j] += ke[1, 1]
        fe = np.zeros(2, dtype=float)
        J = le / 2.0
        xm = 0.5 * (x1 + x2)
        for q in range(gauss_xi.size):
            xi = gauss_xi[q]
            w = gauss_w[q]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            xq = xm + xi * J
            fval = float(body_force_fn(float(xq)))
            fe[0] += N1 * fval * J * w
            fe[1] += N2 * fval * J * w
        F[i] += fe[0]
        F[j] += fe[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            if bc is None:
                continue
            if 'x_location' not in bc or 'load_mag' not in bc:
                raise ValueError("Each Neumann BC must have 'x_location' and 'load_mag'.")
            xloc = float(bc['x_location'])
            load = float(bc['load_mag'])
            idx = np.where(np.isclose(node_coords, xloc, rtol=0.0, atol=tol))[0]
            if idx.size != 1:
                raise ValueError(f'Neumann BC location does not match a unique node: {xloc}')
            F[int(idx[0])] += load
    if dirichlet_bc_list is None:
        dirichlet_bc_list = []
    dir_nodes_list = []
    dir_vals_list = []
    used = {}
    for bc in dirichlet_bc_list:
        if bc is None:
            continue
        if 'x_location' not in bc or 'u_prescribed' not in bc:
            raise ValueError("Each Dirichlet BC must have 'x_location' and 'u_prescribed'.")
        xloc = float(bc['x_location'])
        uval = float(bc['u_prescribed'])
        idx = np.where(np.isclose(node_coords, xloc, rtol=0.0, atol=tol))[0]
        if idx.size != 1:
            raise ValueError(f'Dirichlet BC location does not match a unique node: {xloc}')
        idxi = int(idx[0])
        if idxi in used:
            if not np.isclose(used[idxi], uval, rtol=1e-12, atol=1e-12):
                raise ValueError(f'Conflicting Dirichlet BC values specified at node {idxi}.')
            continue
        used[idxi] = uval
        dir_nodes_list.append(idxi)
        dir_vals_list.append(uval)
    dir_nodes = np.array(dir_nodes_list, dtype=int)
    dir_vals = np.array(dir_vals_list, dtype=float)
    all_nodes = np.arange(n_nodes, dtype=int)
    if dir_nodes.size > 0:
        mask_free = np.ones(n_nodes, dtype=bool)
        mask_free[dir_nodes] = False
        free_nodes = all_nodes[mask_free]
    else:
        free_nodes = all_nodes
    u = np.zeros(n_nodes, dtype=float)
    if dir_nodes.size == 0:
        try:
            u = np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            raise
        reactions = np.zeros(0, dtype=float)
        reaction_nodes = np.array([], dtype=int)
    else:
        K_ff = K[np.ix_(free_nodes, free_nodes)]
        K_fc = K[np.ix_(free_nodes, dir_nodes)]
        F_f = F[free_nodes]
        rhs = F_f - K_fc @ dir_vals
        try:
            u_f = np.linalg.solve(K_ff, rhs)
        except np.linalg.LinAlgError:
            raise
        u[free_nodes] = u_f
        u[dir_nodes] = dir_vals
        R_all = K @ u - F
        reactions = R_all[dir_nodes]
        reaction_nodes = dir_nodes
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}