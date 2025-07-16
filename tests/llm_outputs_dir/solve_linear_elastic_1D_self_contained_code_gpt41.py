import numpy as np
from typing import Callable, List, Dict, Optional

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
            - "displacements" (np.ndarray): Displacement at each node, shape (n_nodes,)
            - "reactions" (np.ndarray): Reaction forces at Dirichlet BC nodes, shape (n_dirichlet,)
            - "node_coords" (np.ndarray): Coordinates of all nodes, shape (n_nodes,)
            - "reaction_nodes" (np.ndarray): Indices of Dirichlet BC nodes, shape (n_dirichlet,)
    """
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    element_node_ids = np.column_stack((np.arange(num_elements), np.arange(1, num_elements+1)))
    
    # Precompute Gaussian quadrature points and weights
    # xi in [-1, 1]
    if n_gauss == 1:
        gauss_xi = np.array([0.0])
        gauss_wt = np.array([2.0])
    elif n_gauss == 2:
        gauss_xi = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
        gauss_wt = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        gauss_wt = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("n_gauss must be 1, 2, or 3")
    
    # Material assignment function, for a given x, returns region E and A
    def mat_props_at_x(x):
        for reg in material_regions:
            if reg["coord_min"] <= x <= reg["coord_max"]:
                return reg["E"], reg["A"]
        raise ValueError(f"x={x} not in any material region")

    K = np.zeros((n_nodes, n_nodes), dtype=float)  # Global stiffness
    F = np.zeros(n_nodes, dtype=float)  # Global force

    # Assembly process
    for e in range(num_elements):
        n1, n2 = element_node_ids[e]
        x1, x2 = node_coords[n1], node_coords[n2]
        le = x2 - x1
        
        # Get element property at midpoint
        x_mid = 0.5 * (x1 + x2)
        E, A = mat_props_at_x(x_mid)
        
        # Local stiffness for linear element: ke = (E*A/le) * [[1, -1], [-1, 1]]
        ke = (E * A / le) * np.array([[1, -1], [-1, 1]], dtype=float)
        
        # Body force vector by Gauss quadrature
        fe = np.zeros(2, dtype=float)
        for k in range(n_gauss):
            xi = gauss_xi[k]
            w = gauss_wt[k]
            # Mapping xi in [-1,1] to x in [x1,x2]
            xg = ((1-xi)/2)*x1 + ((1+xi)/2)*x2
            J = le/2  # dx/dxi
            # Linear shape functions
            N = np.array([(1-xi)/2, (1+xi)/2])
            fval = body_force_fn(xg)
            fe += N * fval * w * J  # f(x) * N * w * jacobian

        # Assemble to global
        K[np.ix_([n1, n2],[n1, n2])] += ke
        F[[n1, n2]] += fe

    # Apply Neumann BC (point loads)
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc["x_location"]
            load_mag = bc["load_mag"]
            # Find closest node
            node_id = np.argmin(np.abs(node_coords - x_loc))
            F[node_id] += load_mag

    # Dirichlet BC
    dirichlet_node_ids = []
    prescribed_u = []
    for bc in dirichlet_bc_list:
        x_loc = bc["x_location"]
        u_val = bc["u_prescribed"]
        node_id = np.argmin(np.abs(node_coords - x_loc))
        dirichlet_node_ids.append(node_id)
        prescribed_u.append(u_val)
    dirichlet_node_ids = np.array(dirichlet_node_ids, dtype=int)
    prescribed_u = np.array(prescribed_u, dtype=float)
    all_dofs = np.arange(n_nodes)
    free_dofs = np.setdiff1d(all_dofs, dirichlet_node_ids)
    
    # Apply Dirichlet BC using elimination (modify K, F)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fc = K[np.ix_(free_dofs, dirichlet_node_ids)]
    F_f = F[free_dofs] - K_fc.dot(prescribed_u)
    # Solve for unknowns
    U = np.zeros(n_nodes, dtype=float)
    U[dirichlet_node_ids] = prescribed_u
    if K_ff.shape[0] > 0:
        U[free_dofs] = np.linalg.solve(K_ff, F_f)
    # Reaction forces
    reactions = K[np.ix_(dirichlet_node_ids, all_dofs)].dot(U) - F[dirichlet_node_ids]
    return {
        "displacements": U,
        "reactions": reactions,
        "node_coords": node_coords,
        "reaction_nodes": dirichlet_node_ids
    }