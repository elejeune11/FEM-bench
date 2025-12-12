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
    import numpy as np
    from typing import Callable, List, Dict, Optional
    import pytest
    if n_gauss not in (1, 2, 3):
        raise ValueError('n_gauss must be 1, 2, or 3.')
    if x_max <= x_min:
        raise ValueError('x_max must be greater than x_min.')
    if num_elements < 1:
        raise ValueError('num_elements must be at least 1.')
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    K = np.zeros((n_nodes, n_nodes), dtype=float)
    F = np.zeros(n_nodes, dtype=float)
    if n_gauss == 1:
        xi_pts = np.array([0.0])
        weights = np.array([2.0])
    elif n_gauss == 2:
        xi_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        weights = np.array([1.0, 1.0])
    else:
        xi_pts = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    for e in range(num_elements):
        n1 = e
        n2 = e + 1
        x1 = float(node_coords[n1])
        x2 = float(node_coords[n2])
        L = x2 - x1
        x_center = 0.5 * (x1 + x2)
        matches = []
        for reg in material_regions:
            cmin = float(reg['coord_min'])
            cmax = float(reg['coord_max'])
            if x_center >= cmin - 1e-12 and x_center <= cmax + 1e-12:
                matches.append(reg)
        if len(matches) != 1:
            raise ValueError('Element at x_center={} not covered by exactly one material region.'.format(x_center))
        region = matches[0]
        E = float(region['E'])
        A = float(region['A'])
        ke = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
        K[n1:n2 + 1, n1:n2 + 1] += ke
        J = L / 2.0
        fe_local = np.zeros(2, dtype=float)
        for (xi, w) in zip(xi_pts, weights):
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            x_gp = N1 * x1 + N2 * x2
            fval = float(body_force_fn(float(x_gp)))
            fe_local += np.array([N1, N2], dtype=float) * fval * w * J
        F[n1] += fe_local[0]
        F[n2] += fe_local[1]
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = float(bc['x_location'])
            load = float(bc['load_mag'])
            idxs = np.where(np.isclose(node_coords, x_loc, atol=1e-08, rtol=0.0))[0]
            if idxs.size != 1:
                raise ValueError('Neumann BC location {} does not match a unique node.'.format(x_loc))
            F[idxs[0]] += load
    ordered_dir_nodes = []
    ordered_dir_vals = []
    unique_dir_map = {}
    for bc in dirichlet_bc_list:
        x_loc = float(bc['x_location'])
        u_val = float(bc['u_prescribed'])
        idxs = np.where(np.isclose(node_coords, x_loc, atol=1e-08, rtol=0.0))[0]
        if idxs.size != 1:
            raise ValueError('Dirichlet BC location {} does not match a unique node.'.format(x_loc))
        idx = int(idxs[0])
        ordered_dir_nodes.append(idx)
        ordered_dir_vals.append(u_val)
        if idx in unique_dir_map:
            if not np.isclose(unique_dir_map[idx], u_val, atol=1e-12, rtol=0.0):
                raise ValueError('Conflicting Dirichlet BCs prescribed at node {}.'.format(idx))
        else:
            unique_dir_map[idx] = u_val
    if len(unique_dir_map) > 0:
        unique_nodes = np.array(list(unique_dir_map.keys()), dtype=int)
        unique_vals = np.array([unique_dir_map[k] for k in unique_nodes], dtype=float)
    else:
        unique_nodes = np.array([], dtype=int)
        unique_vals = np.array([], dtype=float)
    all_nodes = np.arange(n_nodes, dtype=int)
    free_nodes = np.setdiff1d(all_nodes, unique_nodes, assume_unique=True)
    u = np.zeros(n_nodes, dtype=float)
    if free_nodes.size == 0:
        for (idx, val) in zip(unique_nodes, unique_vals):
            u[idx] = val
    else:
        K_ff = K[np.ix_(free_nodes, free_nodes)]
        F_f = F[free_nodes].copy()
        if unique_nodes.size > 0:
            K_fc = K[np.ix_(free_nodes, unique_nodes)]
            rhs = F_f - K_fc.dot(unique_vals)
        else:
            rhs = F_f
        u_f = np.linalg.solve(K_ff, rhs)
        u[free_nodes] = u_f
        for (idx, val) in zip(unique_nodes, unique_vals):
            u[idx] = val
    reactions_full = K.dot(u) - F
    reaction_nodes = np.array(ordered_dir_nodes, dtype=int)
    reactions = np.array([reactions_full[idx] for idx in ordered_dir_nodes], dtype=float)
    return {'displacements': u, 'reactions': reactions, 'node_coords': node_coords, 'reaction_nodes': reaction_nodes}