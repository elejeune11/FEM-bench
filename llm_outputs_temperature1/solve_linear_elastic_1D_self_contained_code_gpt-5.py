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
    if num_elements < 1:
        raise ValueError('num_elements must be >= 1')
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3')
    if n_gauss == 1:
        gauss_xi = np.array([0.0], dtype=float)
        gauss_w = np.array([2.0], dtype=float)
    elif n_gauss == 2:
        xi = 1.0 / np.sqrt(3.0)
        gauss_xi = np.array([-xi, xi], dtype=float)
        gauss_w = np.array([1.0, 1.0], dtype=float)
    else:
        xi = np.sqrt(3.0 / 5.0)
        gauss_xi = np.array([-xi, 0.0, xi], dtype=float)
        gauss_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes, dtype=float)
    tol = 1e-12 * max(1.0, abs(x_max - x_min))

    def EA_and_A_at(x: float) -> (float, float):
        for reg in material_regions:
            cmin = float(reg['coord_min'])
            cmax = float(reg['coord_max'])
            if x >= cmin - tol and x <= cmax + tol:
                E = float(reg['E'])
                A = float(reg['A'])
                return (E * A, A)
        raise ValueError(f'No material region found covering x = {x}')
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    for e in range(num_elements):
        i1 = e
        i2 = e + 1
        x1 = node_coords[i1]
        x2 = node_coords[i2]
        L = x2 - x1
        if L <= 0.0:
            raise ValueError('Non-positive element length encountered; check x_min, x_max, and mesh.')
        sum_w_EA = 0.0
        f_e = np.zeros(2, dtype=float)
        for (w, xi) in zip(gauss_w, gauss_xi):
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            xg = N1 * x1 + N2 * x2
            (EA_g, A_g) = EA_and_A_at(xg)
            sum_w_EA += w * EA_g
            f_body = float(body_force_fn(float(xg)))
            f_e += np.array([N1, N2], dtype=float) * (A_g * f_body) * (L * 0.5) * w
        ke_coeff = sum_w_EA / (2.0 * L)
        k_e = ke_coeff * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        K[i1, i1] += k_e[0, 0]
        K[i1, i2] += k_e[0, 1]
        K[i2, i1] += k_e[1, 0]
        K[i2, i2] += k_e[1, 1]
        F[i1] += f_e[0]
        F[i2] += f_e[1]
    if neumann_bc_list:
        for bc in neumann_bc_list:
            xloc = float(bc['x_location'])
            load = float(bc['load_mag'])
            diffs = np.abs(node_coords - xloc)
            idx_candidates = np.where(diffs <= tol)[0]
            if idx_candidates.size == 0:
                raise ValueError(f'Neumann BC location {xloc} does not match any node.')
            idx = int(idx_candidates[0])
            F[idx] += load
    if not dirichlet_bc_list or len(dirichlet_bc_list) == 0:
        raise ValueError('At least one Dirichlet boundary condition is required for a unique solution.')
    bc_nodes_list = []
    bc_vals_list = []
    seen = {}
    for bc in dirichlet_bc_list:
        xloc = float(bc['x_location'])
        uval = float(bc['u_prescribed'])
        diffs = np.abs(node_coords - xloc)
        idx_candidates = np.where(diffs <= tol)[0]
        if idx_candidates.size == 0:
            raise ValueError(f'Dirichlet BC location {xloc} does not match any node.')
        idx = int(idx_candidates[0])
        if idx in seen:
            if not np.isclose(seen[idx], uval):
                raise ValueError(f'Conflicting Dirichlet BC values at node {idx}: {seen[idx]} vs {uval}')
            continue
        seen[idx] = uval
        bc_nodes_list.append(idx)
        bc_vals_list.append(uval)
    reaction_nodes = np.array(bc_nodes_list, dtype=int)
    u = np.zeros(n_nodes, dtype=float)
    u[reaction_nodes] = np.array(bc_vals_list, dtype=float)
    all_nodes = np.arange(n_nodes, dtype=int)
    free_dofs = np.setdiff1d(all_nodes, reaction_nodes, assume_unique=False)
    if free_dofs.size > 0:
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        K_fc = K[np.ix_(free_dofs, reaction_nodes)]
        rhs = F[free_dofs] - K_fc @ u[reaction_nodes]
        try:
            u[free_dofs] = np.linalg.solve(K_ff, rhs)
        except np.linalg.LinAlgError as e:
            raise ValueError('Global stiffness matrix for free DOFs is singular. Check boundary conditions.') from e
    residual = K @ u - F
    reactions = residual[reaction_nodes]
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}