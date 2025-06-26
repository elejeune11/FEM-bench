import numpy as np
from typing import Callable, List, Dict


def linear_uniform_mesh_1D(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    """
    Generate a 1D linear mesh.

    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.

    Returns:
        np.ndarray[np.ndarray, np.ndarray]:
            - node_coords: 1D array of node coordinates (shape: [num_nodes])
            - element_connectivity: 2D array of element connectivity 
              (shape: [num_elements, 2]) with node indices per element
    """
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_connectivity = np.zeros((num_elements, 2), dtype=int)

    for e in range(num_elements):
        element_connectivity[e, 0] = e
        element_connectivity[e, 1] = e + 1

    return np.array([node_coords, element_connectivity], dtype=object)


def shape_functions_1D_linear(xi: float) -> np.ndarray:
    """
    Compute 1D linear shape functions N1 and N2 at a given ξ ∈ [-1, 1].

    Parameters:
        xi (float): Reference coordinate ξ

    Returns:
        np.ndarray: Shape function values [N1(ξ), N2(ξ)]
    """
    return np.array([
        0.5 * (1 - xi),
        0.5 * (1 + xi)
    ])


def shape_function_derivatives_1D_linear() -> np.ndarray:
    """
    Return constant derivatives of 1D linear shape functions w.r.t. ξ.

    Returns:
        np.ndarray: Derivatives [dN1/dξ, dN2/dξ]
    """
    return np.array([
        -0.5,
        0.5
    ])


def compute_jacobian_1D(dN_dxi: np.ndarray, x_elem: np.ndarray) -> float:
    """
    Compute the Jacobian for 1D isoparametric mapping.

    Parameters:
        dN_dxi (np.ndarray): Shape function derivatives w.r.t. ξ
        x_elem (np.ndarray): Nodal coordinates of the current element [x1, x2]

    Returns:
        float: Jacobian dx/dξ
    """
    return np.array(np.dot(dN_dxi, x_elem))


def gauss_quadrature_1D(n: int) -> np.ndarray:
    """
    Return Gauss points and weights for 1D quadrature over [-1, 1].

    Parameters:
        n (int): Number of Gauss points (1, 2, or 3 recommended)

    Returns:
        np.ndarray[np.ndarray, np.ndarray]: (points, weights)
    """
    if n == 1:
        points = np.array([0.0])
        weights = np.array([2.0])
    elif n == 2:
        sqrt_3_inv = 1.0 / np.sqrt(3)
        points = np.array([-sqrt_3_inv, sqrt_3_inv])
        weights = np.array([1.0, 1.0])
    elif n == 3:
        points = np.array([
            -np.sqrt(3/5), 0.0, np.sqrt(3/5)
        ])
        weights = np.array([
            5/9, 8/9, 5/9
        ])
    else:
        raise ValueError("Only 1 to 3 Gauss points are supported.")
    
    return np.array([points, weights], dtype=object)


def assign_element_props_linear_elastic_1D(
    prop_list: List[Dict[str, float]],
    node_coords: np.ndarray,
    element_connectivity: np.ndarray
) -> np.ndarray:
    """
    Assign per-element linear elastic properties (E, A) for multiple material regions.

    Parameters:
        prop_list (List[Dict]): Each dict must contain:
            {
                "coord_min": float,
                "coord_max": float,
                "E": float,
                "A": float
            }
        node_coords (np.ndarray): Coordinates of all nodes, shape (n_nodes,)
        element_connectivity (np.ndarray): Element connectivity (n_elements, 2)

    Returns:
        np.ndarray[np.ndarray, np.ndarray]: E_vector, A_vector (both shape: n_elements,)
    """
    n_elements = element_connectivity.shape[0]
    E_vector = np.zeros(n_elements)
    A_vector = np.zeros(n_elements)

    for i in range(n_elements):
        node_indices = element_connectivity[i]
        x1, x2 = node_coords[node_indices[0]], node_coords[node_indices[1]]
        x_center = 0.5 * (x1 + x2)

        assigned = False
        for region in prop_list:
            if region["coord_min"] <= x_center <= region["coord_max"]:
                E_vector[i] = region["E"]
                A_vector[i] = region["A"]
                assigned = True
                break  # First match wins

        if not assigned:
            raise ValueError(f"Element {i} centered at x = {x_center:.3f} is not assigned to any material region.")

    return np.array([E_vector, A_vector], dtype=object)


def element_stiffness_linear_elastic_1D(
    x_elem: np.ndarray,  # nodal coordinates [x1, x2]
    E: float,
    A: float,
    n_gauss: int = 2
) -> np.ndarray:
    """
    Compute the element stiffness matrix for a 1D linear bar using the Galerkin method.

    Parameters:
        x_elem (np.ndarray): Nodal coordinates of the element [x1, x2]
        E (float): Young's modulus
        A (float): Cross-sectional area
        n_gauss (int): Number of Gauss integration points (default = 2)

    Returns:
        np.ndarray: 2x2 element stiffness matrix
    """
    k_elem = np.zeros((2, 2))
    xi_points, weights = gauss_quadrature_1D(n_gauss)
    dN_dxi = shape_function_derivatives_1D_linear()

    for xi, w in zip(xi_points, weights):
        J = compute_jacobian_1D(dN_dxi, x_elem)
        dN_dx = dN_dxi / J  # physical shape function derivatives
        B = dN_dx.reshape(1, 2)  # row vector

        # k = ∫ Bᵀ * EA * B * J dξ ≈ Σ Bᵀ * EA * B * J * w
        k_elem += E * A * J * w * (B.T @ B)

    return k_elem


def element_body_force_vector_1D(
    x_elem: np.ndarray,
    body_force_fn,
    n_gauss: int = 2
) -> np.ndarray:
    """
    Compute element body force vector using shape functions and isoparametric mapping.

    Parameters:
        x_elem (np.ndarray): Coordinates of element nodes [x1, x2]
        body_force_fn (callable): Function f(x) for distributed body force
        n_gauss (int): Number of Gauss points

    Returns:
        np.ndarray: Element body force vector (2,)
    """
    fe = np.zeros(2)
    xi_points, weights = gauss_quadrature_1D(n_gauss)
    dN_dxi = shape_function_derivatives_1D_linear()

    for xi, w in zip(xi_points, weights):
        N = shape_functions_1D_linear(xi)
        J = compute_jacobian_1D(dN_dxi, x_elem)
        x_phys = np.dot(N, x_elem)
        f = body_force_fn(x_phys)
        fe += w * f * J * N  # N is shape (2,), f is scalar

    return fe


def assemble_global_stiffness_matrix_linear_elastic_1D(
    node_coords: np.ndarray,
    element_connectivity: np.ndarray,
    E_vector: np.ndarray,
    A_vector: np.ndarray,
    n_gauss: int = 2
) -> np.ndarray:
    """
    Assemble the global stiffness matrix for a 1D linear elastic bar.

    Parameters:
        node_coords (np.ndarray): 1D array of node coordinates (n_nodes,)
        element_connectivity (np.ndarray): 2D array of shape (n_elements, 2)
        E_vector (np.ndarray): Young's modulus for each element (n_elements,)
        A_vector (np.ndarray): Cross-sectional area for each element (n_elements,)
        n_gauss (int): Number of Gauss integration points per element

    Returns:
        np.ndarray: Global stiffness matrix of shape (n_nodes, n_nodes)
    """
    n_nodes = len(node_coords)
    n_elements = element_connectivity.shape[0]
    K_global = np.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        node_ids = element_connectivity[e]
        x_elem = node_coords[node_ids]
        E = E_vector[e]
        A = A_vector[e]

        # Compute element stiffness matrix using Galerkin method
        k_elem = element_stiffness_linear_elastic_1D(x_elem, E, A, n_gauss)

        # Assemble into global matrix
        for a_local, a_global in enumerate(node_ids):
            for b_local, b_global in enumerate(node_ids):
                K_global[a_global, b_global] += k_elem[a_local, b_local]

    return K_global


def assemble_global_body_force_vector_linear_elastic_1D(
    node_coords: np.ndarray,
    element_connectivity: np.ndarray,
    body_force_fn: Callable[[float], float],
    n_gauss: int = 2
) -> np.ndarray:
    """
    Assemble the global body force vector for a 1D linear elastic bar.

    Parameters:
        node_coords (np.ndarray): Node coordinates (n_nodes,)
        element_connectivity (np.ndarray): Connectivity matrix (n_elements, 2)
        body_force_fn (Callable): Function f(x) defining the body force
        n_gauss (int): Number of Gauss points per element

    Returns:
        np.ndarray: Global body force vector (n_nodes,)
    """
    n_nodes = len(node_coords)
    n_elements = element_connectivity.shape[0]
    F_global = np.zeros(n_nodes)

    for e in range(n_elements):
        node_ids = element_connectivity[e]
        x_elem = node_coords[node_ids]

        # Compute the element body force vector
        f_elem = element_body_force_vector_1D(x_elem, body_force_fn, n_gauss)

        # Assemble into the global force vector
        for a_local, a_global in enumerate(node_ids):
            F_global[a_global] += f_elem[a_local]

    return F_global


def solve_matrix_eqn(
    K_global: np.ndarray,
    F_global: np.ndarray,
    dirichlet_BC_nodes: List[int],
    prescribed_displacements: np.ndarray
) -> np.ndarray:
    """
    Solve the global system with Dirichlet boundary conditions.

    Parameters:
        K_global (np.ndarray): Global stiffness matrix (n_nodes, n_nodes)
        F_global (np.ndarray): Global force vector (n_nodes,)
        dirichlet_BC_nodes (List[int]): Indices of nodes with prescribed displacements
        prescribed_displacements (np.ndarray): Displacement values at Dirichlet nodes (same length)

    Returns:
        np.ndarray:
            x (np.ndarray): Full displacement vector (n_nodes,)
            R (np.ndarray): Reaction forces at Dirichlet (supported) nodes (same order as input list)
    """
    # Partition system
    n_nodes = K_global.shape[0]
    all_nodes = set(range(n_nodes))
    free_nodes = sorted(all_nodes - set(dirichlet_BC_nodes))
    supported_nodes = sorted(dirichlet_BC_nodes)

    # Convert to index arrays
    f = np.array(free_nodes)
    s = np.array(supported_nodes)

    # Partition stiffness matrix and force vector
    K_ff = K_global[np.ix_(f, f)]
    K_ss = K_global[np.ix_(s, s)]
    K_fs = K_global[np.ix_(f, s)]
    K_sf = K_global[np.ix_(s, f)]
    F_f = F_global[f]
    F_s = F_global[s]

    # Solve for unknown displacements at free nodes
    x_s = np.array(prescribed_displacements)
    x_f = np.linalg.solve(K_ff, F_f - K_fs @ x_s)

    # Build full displacement vector
    x = np.zeros(n_nodes)
    for idx, i in enumerate(free_nodes):
        x[i] = x_f[idx]
    for idx, i in enumerate(dirichlet_BC_nodes):
        x[i] = x_s[idx]

    # Compute reaction forces at supported nodes
    R = K_sf @ x_f + K_ss @ x_s - F_s

    return np.array([x, R], dtype=object)


def solve_linear_elastic_1D(
    node_coords: np.ndarray,
    element_connectivity: np.ndarray,
    prop_list: List[Dict[str, float]],
    body_force_fn: Callable[[float], float],
    dirichlet_BC_locations: List[float],
    prescribed_displacements: List[float],
    neumann_bc_list: List[Dict[str, float]] = None,
    n_gauss: int = 2
) -> np.ndarray:
    """
    Solve a 1D linear elastic finite element problem.

    Parameters:
        node_coords (np.ndarray): Node coordinates (n_nodes,)
        element_connectivity (np.ndarray): Element connectivity (n_elements, 2)
        prop_list (List[Dict]): List of material regions, each with keys:
            "coord_min", "coord_max", "E", "A"
        body_force_fn (Callable): Function f(x) for body force
        dirichlet_BC_locations (List[float]): Positions where displacements are prescribed
        prescribed_displacements (List[float]): Values of displacements at those positions
        neumann_bc_list (List[Dict]): Each dict must contain:
            {
                "x_location": float,  # coordinate of the node
                "load_mag": float     # magnitude of point load (positive = outward)
            }
        n_gauss (int): Number of Gauss points for numerical integration

    Returns:
        np.ndarray:
            - x (np.ndarray): Displacement vector (n_nodes,)
            - R (np.ndarray): Reaction forces at Dirichlet BC nodes
    """
    # Assign material properties to elements
    E_vector, A_vector = assign_element_props_linear_elastic_1D(
        prop_list, node_coords, element_connectivity
    )

    # Assemble global stiffness matrix
    K_global = assemble_global_stiffness_matrix_linear_elastic_1D(
        node_coords, element_connectivity, E_vector, A_vector, n_gauss
    )

    # Assemble global body force vector
    F_global = assemble_global_body_force_vector_linear_elastic_1D(
        node_coords, element_connectivity, body_force_fn, n_gauss
    )

    # Apply Neumann boundary conditions (point loads)
    tol = 1e-8
    if neumann_bc_list is not None:
        for bc in neumann_bc_list:
            x_target = bc["x_location"]
            load = bc["load_mag"]
            
            matches = np.where(np.abs(node_coords - x_target) < tol)[0]
            
            if len(matches) == 0:
                raise ValueError(f"No node found at x = {x_target} within tolerance {tol}")
            elif len(matches) > 1:
                raise ValueError(f"Multiple nodes found near x = {x_target}; check mesh or tolerance.")
            
            F_global[matches[0]] += load

    # Map Dirichlet BC locations to node indices
    dirichlet_BC_nodes = []
    for x_target in dirichlet_BC_locations:
        matches = np.where(np.abs(node_coords - x_target) < tol)[0]
        
        if len(matches) == 0:
            raise ValueError(f"No node found at x = {x_target} within tolerance {tol}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple nodes found near x = {x_target}; check mesh spacing or reduce tolerance.")
        
        dirichlet_BC_nodes.append(matches[0])

    # Solve for displacements and reactions
    x, R = solve_matrix_eqn(
        K_global, F_global, dirichlet_BC_nodes, prescribed_displacements
    )

    return np.array([x, R], dtype=object)


def solve_linear_elastic_1D_self_contained(
    x_min: float,
    x_max: float,
    num_elements: int,
    prop_list: List[Dict[str, float]],
    body_force_fn: Callable[[float], float],
    dirichlet_BC_locations: List[float],
    prescribed_displacements: List[float],
    neumann_bc_list: List[Dict[str, float]] = None,
    n_gauss: int = 2
) -> np.ndarray:
    """
    Solve a 1D linear elastic finite element problem with integrated meshing.

    Parameters:
        x_min (float): Start coordinate of the domain
        x_max (float): End coordinate of the domain
        num_elements (int): Number of linear elements
        prop_list (List[Dict]): List of material regions, each with keys:
            "coord_min", "coord_max", "E", "A"
        body_force_fn (Callable): Function f(x) for body force
        dirichlet_BC_locations (List[float]): Positions where displacements are prescribed
        prescribed_displacements (List[float]): Values of displacements at those positions
        neumann_bc_list (List[Dict]): Each dict must contain:
            {
                "x_location": float,  # coordinate of the node
                "load_mag": float     # magnitude of point load (positive = outward)
            }
        n_gauss (int): Number of Gauss points for numerical integration

    Returns:
        np.ndarray[np.ndarray, np.ndarray]:
            - x (np.ndarray): Displacement vector (n_nodes,)
            - R (np.ndarray): Reaction forces at Dirichlet BC nodes
    """
    
    # Step 0: Generate mesh
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_connectivity = np.zeros((num_elements, 2), dtype=int)
    
    for e in range(num_elements):
        element_connectivity[e, 0] = e
        element_connectivity[e, 1] = e + 1
    
    n_nodes = len(node_coords)
    n_elements = element_connectivity.shape[0]
    
    # Step 1: Assign material properties to elements
    E_vector = np.zeros(n_elements)
    A_vector = np.zeros(n_elements)
    
    for i in range(n_elements):
        node_indices = element_connectivity[i]
        x1, x2 = node_coords[node_indices[0]], node_coords[node_indices[1]]
        x_center = 0.5 * (x1 + x2)
        
        assigned = False
        for region in prop_list:
            if region["coord_min"] <= x_center <= region["coord_max"]:
                E_vector[i] = region["E"]
                A_vector[i] = region["A"]
                assigned = True
                break
        
        if not assigned:
            raise ValueError(f"Element {i} centered at x = {x_center:.3f} is not assigned to any material region.")
    
    # Step 2: Set up Gauss quadrature points and weights
    if n_gauss == 1:
        xi_points = np.array([0.0])
        weights = np.array([2.0])
    elif n_gauss == 2:
        sqrt_3_inv = 1.0 / np.sqrt(3)
        xi_points = np.array([-sqrt_3_inv, sqrt_3_inv])
        weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        xi_points = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("Only 1 to 3 Gauss points are supported.")
    
    # Shape function derivatives (constant for linear elements)
    dN_dxi = np.array([-0.5, 0.5])
    
    # Step 3: Assemble global stiffness matrix
    K_global = np.zeros((n_nodes, n_nodes))
    
    for e in range(n_elements):
        node_ids = element_connectivity[e]
        x_elem = node_coords[node_ids]
        E = E_vector[e]
        A = A_vector[e]
        
        # Compute element stiffness matrix
        k_elem = np.zeros((2, 2))
        
        for xi, w in zip(xi_points, weights):
            # Jacobian
            J = np.dot(dN_dxi, x_elem)
            
            # Physical derivatives
            dN_dx = dN_dxi / J
            B = dN_dx.reshape(1, 2)
            
            # Add contribution: k = ∫ Bᵀ * EA * B * J dξ
            k_elem += E * A * J * w * (B.T @ B)
        
        # Assemble into global matrix
        for a_local, a_global in enumerate(node_ids):
            for b_local, b_global in enumerate(node_ids):
                K_global[a_global, b_global] += k_elem[a_local, b_local]
    
    # Step 4: Assemble global body force vector
    F_global = np.zeros(n_nodes)
    
    for e in range(n_elements):
        node_ids = element_connectivity[e]
        x_elem = node_coords[node_ids]
        
        # Compute element body force vector
        fe = np.zeros(2)
        
        for xi, w in zip(xi_points, weights):
            # Shape functions at xi
            N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
            
            # Jacobian
            J = np.dot(dN_dxi, x_elem)
            
            # Physical coordinate
            x_phys = np.dot(N, x_elem)
            
            # Body force at physical coordinate
            f = body_force_fn(x_phys)
            
            # Add contribution
            fe += w * f * J * N
        
        # Assemble into global force vector
        for a_local, a_global in enumerate(node_ids):
            F_global[a_global] += fe[a_local]
    
    # Step 5: Apply Neumann boundary conditions (point loads)
    if neumann_bc_list is not None:
        tol = 1e-8
        for bc in neumann_bc_list:
            x_target = bc["x_location"]
            load = bc["load_mag"]
            
            matches = np.where(np.abs(node_coords - x_target) < tol)[0]
            
            if len(matches) == 0:
                raise ValueError(f"No node found at x = {x_target} within tolerance {tol}")
            elif len(matches) > 1:
                raise ValueError(f"Multiple nodes found near x = {x_target}; check mesh or tolerance.")
            
            node_index = matches[0]
            F_global[node_index] += load
    
    # Step 6: Find node indices for Dirichlet boundary conditions
    dirichlet_BC_nodes = []
    tol = 1e-8
    
    for x_target in dirichlet_BC_locations:
        matches = np.where(np.abs(node_coords - x_target) < tol)[0]
        
        if len(matches) == 0:
            raise ValueError(f"No node found at x = {x_target} within tolerance {tol}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple nodes found near x = {x_target}; check mesh spacing or reduce tolerance.")
        
        dirichlet_BC_nodes.append(matches[0])
    
    # Step 7: Partition system
    all_nodes = set(range(n_nodes))
    free_nodes = sorted(all_nodes - set(dirichlet_BC_nodes))
    supported_nodes = sorted(dirichlet_BC_nodes)
    
    # Convert to index arrays
    f = np.array(free_nodes)
    s = np.array(supported_nodes)
    
    # Partition stiffness matrix
    K_ff = K_global[np.ix_(f, f)]
    K_ss = K_global[np.ix_(s, s)]
    K_fs = K_global[np.ix_(f, s)]
    K_sf = K_global[np.ix_(s, f)]
    
    # Partition force vector
    F_f = F_global[f]
    F_s = F_global[s]
    
    # Step 8: Solve for displacements and reactions
    x_s = np.array(prescribed_displacements)
    x_f = np.linalg.solve(K_ff, F_f - K_fs @ x_s)
    
    # Build full displacement vector
    x = np.zeros(n_nodes)
    for idx, i in enumerate(free_nodes):
        x[i] = x_f[idx]
    for idx, i in enumerate(dirichlet_BC_nodes):
        x[i] = x_s[idx]
    
    # Compute reaction forces at supported nodes
    R = K_sf @ x_f + K_ss @ x_s - F_s
    
    return np.array([x, R], dtype=object)