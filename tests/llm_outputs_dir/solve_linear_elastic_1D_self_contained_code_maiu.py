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
    # Generate node coordinates uniformly
    node_coords = np.linspace(x_min, x_max, num_elements + 1)
    n_nodes = len(node_coords)

    # Connectivity: each element connects node i to i+1
    connectivity = np.array([[i, i+1] for i in range(num_elements)])

    # Assign material properties per element based on material_regions
    E_elems = np.zeros(num_elements)
    A_elems = np.zeros(num_elements)
    for e in range(num_elements):
        x_left = node_coords[connectivity[e,0]]
        x_right = node_coords[connectivity[e,1]]
        x_center = 0.5*(x_left + x_right)
        for region in material_regions:
            if region["coord_min"] <= x_center <= region["coord_max"]:
                E_elems[e] = region["E"]
                A_elems[e] = region["A"]
                break

    # Gauss points and weights for 1D [-1,1]
    if n_gauss == 1:
        gauss_pts = np.array([0.0])
        gauss_wts = np.array([2.0])
    elif n_gauss == 2:
        gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        gauss_wts = np.array([1.0, 1.0])
    elif n_gauss == 3:
        gauss_pts = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        gauss_wts = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("n_gauss must be 1, 2 or 3")

    # Initialize global stiffness matrix and force vector
    K = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    # Shape functions and derivatives in reference coords xi in [-1,1]
    def N(xi):
        return np.array([(1 - xi)/2, (1 + xi)/2])
    def dN_dxi(xi):
        return np.array([-0.5, 0.5])

    # Assembly over elements
    for e in range(num_elements):
        nodes = connectivity[e]
        x_e = node_coords[nodes]
        h_e = x_e[1] - x_e[0]
        E_e = E_elems[e]
        A_e = A_elems[e]

        # Element stiffness matrix ke and force vector fe
        ke = np.zeros((2,2))
        fe = np.zeros(2)

        for q in range(n_gauss):
            xi_q = gauss_pts[q]
            w_q = gauss_wts[q]

            # Jacobian dx/dxi
            dx_dxi = h_e / 2
            detJ = dx_dxi

            # Derivative of shape functions wrt x
            dN_dx = dN_dxi(xi_q) / detJ

            # Compute x at xi_q by interpolation
            N_q = N(xi_q)
            x_q = N_q @ x_e

            # Evaluate body force at x_q
            f_q = body_force_fn(x_q)

            # Integrate stiffness matrix
            ke += (E_e * A_e) * np.outer(dN_dx, dN_dx) * detJ * w_q

            # Integrate force vector
            fe += N_q * f_q * detJ * w_q

        # Assemble into global K and F
        for a_local in range(2):
            a_global = nodes[a_local]
            F[a_global] += fe[a_local]
            for b_local in range(2):
                b_global = nodes[b_local]
                K[a_global,b_global] += ke[a_local,b_local]

    # Apply Neumann BCs (point loads)
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_loc = bc["x_location"]
            load_mag = bc["load_mag"]
            # Find closest node index to x_loc
            idx = np.argmin(np.abs(node_coords - x_loc))
            F[idx] += load_mag

    # Apply Dirichlet BCs using penalty or elimination method
    prescribed_nodes = []
    prescribed_values = []
    for bc in dirichlet_bc_list:
        x_loc = bc["x_location"]
        u_val = bc["u_prescribed"]
        idx = np.argmin(np.abs(node_coords - x_loc))
        prescribed_nodes.append(idx)
        prescribed_values.append(u_val)
    prescribed_nodes = np.array(prescribed_nodes)
    prescribed_values = np.array(prescribed_values)

    free_nodes = np.setdiff1d(np.arange(n_nodes), prescribed_nodes)

    # Modify system to apply Dirichlet BCs by elimination
    K_ff = K[np.ix_(free_nodes, free_nodes)]
    K_fc = K[np.ix_(free_nodes, prescribed_nodes)]
    F_f = F[free_nodes] - K_fc @ prescribed_values

    # Solve for unknown displacements
    d_f = np.linalg.solve(K_ff, F_f)

    # Construct full displacement vector
    displacements = np.zeros(n_nodes)
    displacements[free_nodes] = d_f
    displacements[prescribed_nodes] = prescribed_values

    # Compute reactions at Dirichlet nodes: R_c = K_cf d_f + K_cc d_c - F_c
    K_cf = K[np.ix_(prescribed_nodes, free_nodes)]
    K_cc = K[np.ix_(prescribed_nodes, prescribed_nodes)]
    F_c = F[prescribed_nodes]
    reactions = K_cf @ d_f + K_cc @ prescribed_values - F_c

    return {
        "displacements": displacements,
        "reactions": reactions,
        "node_coords": node_coords,
        "reaction_nodes": prescribed_nodes
    }