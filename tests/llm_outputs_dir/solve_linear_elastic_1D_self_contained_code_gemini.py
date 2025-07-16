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
    element_size = (x_max - x_min) / num_elements

    K = np.zeros((n_nodes, n_nodes))
    f = np.zeros(n_nodes)

    # Element stiffness and force assembly
    for i in range(num_elements):
        x1 = node_coords[i]
        x2 = node_coords[i+1]
        element_E = 0
        element_A = 0
        for region in material_regions:
            if region["coord_min"] <= x1 and x2 <= region["coord_max"]:
                element_E = region["E"]
                element_A = region["A"]
                break
            elif region["coord_min"] <= x1 and x1 < region["coord_max"] and x2 > region["coord_max"]:
                element_E = region["E"]
                element_A = region["A"]
                break
            elif region["coord_min"] > x1 and x2 >= region["coord_max"]:
                element_E = region["E"]
                element_A = region["A"]
                break
            elif region["coord_min"] < x2 and x2 <= region["coord_max"] and x1 < region["coord_min"]:
                element_E = region["E"]
                element_A = region["A"]
                break
        
        ke = (element_E * element_A) / element_size * np.array([[1, -1], [-1, 1]])
        K[i:i+2, i:i+2] += ke
        
        # Body force contribution
        if n_gauss == 1:
            gauss_point = (x1 + x2) / 2
            weight = element_size
            fe = body_force_fn(gauss_point) * weight * np.array([0.5, 0.5])
            f[i:i+2] += fe
        elif n_gauss == 2:
            gauss_points = np.array([x1 + element_size * (0.5 - 1/np.sqrt(3)), x1 + element_size * (0.5 + 1/np.sqrt(3))])
            weights = np.array([element_size/2, element_size/2])
            fe = np.zeros(2)
            for j in range(2):
                fe += body_force_fn(gauss_points[j]) * weights[j] * np.array([0.5, 0.5])
            f[i:i+2] += fe
        elif n_gauss == 3:
            gauss_points = np.array([x1 + element_size * (0.5 - np.sqrt(3/5)), x1 + element_size * 0.5, x1 + element_size * (0.5 + np.sqrt(3/5))])
            weights = np.array([5/18 * element_size, 8/18 * element_size, 5/18 * element_size])
            fe = np.zeros(2)
            for j in range(3):
                fe += body_force_fn(gauss_points[j]) * weights[j] * np.array([0.5, 0.5])
            f[i:i+2] += fe

    # Neumann BCs
    if neumann_bc_list:
        for bc in neumann_bc_list:
            node_index = np.argmin(np.abs(node_coords - bc["x_location"]))
            f[node_index] += bc["load_mag"]

    # Apply Dirichlet BCs
    dirichlet_nodes = []
    u_prescribed_values = []
    for bc in dirichlet_bc_list:
        node_index = np.argmin(np.abs(node_coords - bc["x_location"]))
        dirichlet_nodes.append(node_index)
        u_prescribed_values.append(bc["u_prescribed"])
        
        # Modify K and f to incorporate Dirichlet BC
        K[node_index, :] = 0
        K[:, node_index] = 0
        K[node_index, node_index] = 1
        f[node_index] = bc["u_prescribed"]
    
    # Solve for displacements
    displacements = np.linalg.solve(K, f)

    # Calculate reactions
    reactions = np.zeros(len(dirichlet_nodes))
    for i, node_index in enumerate(dirichlet_nodes):
        reactions[i] = np.sum(K[node_index, :] * displacements)

    return {
        "displacements": displacements,
        "reactions": reactions,
        "node_coords": node_coords,
        "reaction_nodes": np.array(dirichlet_nodes)
    }