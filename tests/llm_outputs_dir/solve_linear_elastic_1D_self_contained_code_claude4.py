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
    # Generate mesh
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)
    
    # Gauss quadrature points and weights
    gauss_data = {
        1: (np.array([0.0]), np.array([2.0])),
        2: (np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)]), np.array([1.0, 1.0])),
        3: (np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)]), np.array([5.0/9.0, 8.0/9.0, 5.0/9.0]))
    }
    xi_gauss, w_gauss = gauss_data[n_gauss]
    
    # Initialize global stiffness matrix and force vector
    K_global = np.zeros((n_nodes, n_nodes))
    f_global = np.zeros(n_nodes)
    
    # Assemble element contributions
    for e in range(num_elements):
        # Element nodes
        n1, n2 = e, e + 1
        x1, x2 = node_coords[n1], node_coords[n2]
        L_e = x2 - x1
        
        # Find material properties for this element
        x_mid = (x1 + x2) / 2
        E, A = None, None
        for region in material_regions:
            if region["coord_min"] <= x_mid <= region["coord_max"]:
                E = region["E"]
                A = region["A"]
                break
        
        # Element stiffness matrix
        K_e = np.zeros((2, 2))
        f_e = np.zeros(2)
        
        # Gauss integration
        for i in range(n_gauss):
            xi = xi_gauss[i]
            w = w_gauss[i]
            
            # Shape functions and derivatives
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            dN1_dxi = -0.5
            dN2_dxi = 0.5
            
            # Jacobian
            J = L_e / 2
            
            # Derivatives in physical space
            dN1_dx = dN1_dxi / J
            dN2_dx = dN2_dxi / J
            
            # Physical coordinate
            x_phys = x1 + (xi + 1) * L_e / 2
            
            # Stiffness matrix contribution
            B = np.array([dN1_dx, dN2_dx])
            K_e += E * A * np.outer(B, B) * w * J
            
            # Force vector contribution (body force)
            N = np.array([N1, N2])
            f_body = body_force_fn(x_phys)
            f_e += f_body * A * N * w * J
        
        # Assemble into global system
        dofs = [n1, n2]
        for i in range(2):
            for j in range(2):
                K_global[dofs[i], dofs[j]] += K_e[i, j]
            f_global[dofs[i]] += f_e[i]
    
    # Apply Neumann boundary conditions
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc["x_location"]
            load_mag = bc["load_mag"]
            
            # Find closest node
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            f_global[node_idx] += load_mag
    
    # Apply Dirichlet boundary conditions
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        x_loc = bc["x_location"]
        u_prescribed = bc["u_prescribed"]
        
        # Find closest node
        node_idx = np.argmin(np.abs(node_coords - x_loc))
        reaction_nodes.append(node_idx)
        
        # Modify system for prescribed displacement
        # Zero out row and column, set diagonal to 1, set RHS to prescribed value
        K_global[node_idx, :] = 0
        K_global[:, node_idx] = 0
        K_global[node_idx, node_idx] = 1
        f_global[node_idx] = u_prescribed
    
    reaction_nodes = np.array(reaction_nodes)
    
    # Solve system
    displacements = np.linalg.solve(K_global, f_global)
    
    # Calculate reactions
    reactions = np.zeros(len(reaction_nodes))
    
    # Reassemble original system to calculate reactions
    K_orig = np.zeros((n_nodes, n_nodes))
    f_orig = np.zeros(n_nodes)
    
    for e in range(num_elements):
        n1, n2 = e, e + 1
        x1, x2 = node_coords[n1], node_coords[n2]
        L_e = x2 - x1
        
        x_mid = (x1 + x2) / 2
        E, A = None, None
        for region in material_regions:
            if region["coord_min"] <= x_mid <= region["coord_max"]:
                E = region["E"]
                A = region["A"]
                break
        
        K_e = np.zeros((2, 2))
        f_e = np.zeros(2)
        
        for i in range(n_gauss):
            xi = xi_gauss[i]
            w = w_gauss[i]
            
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            dN1_dxi = -0.5
            dN2_dxi = 0.5
            
            J = L_e / 2
            dN1_dx = dN1_dxi / J
            dN2_dx = dN2_dxi / J
            
            x_phys = x1 + (xi + 1) * L_e / 2
            
            B = np.array([dN1_dx, dN2_dx])
            K_e += E * A * np.outer(B, B) * w * J
            
            N = np.array([N1, N2])
            f_body = body_force_fn(x_phys)
            f_e += f_body * A * N * w * J
        
        dofs = [n1, n2]
        for i in range(2):
            for j in range(2):
                K_orig[dofs[i], dofs[j]] += K_e[i, j]
            f_orig[dofs[i]] += f_e[i]
    
    if neumann_bc_list:
        for bc in neumann_bc_list:
            x_loc = bc["x_location"]
            load_mag = bc["load_mag"]
            node_idx = np.argmin(np.abs(node_coords - x_loc))
            f_orig[node_idx] += load_mag
    
    # Calculate reactions
    for i, node_idx in enumerate(reaction_nodes):
        reactions[i] = np.dot(K_orig[node_idx, :], displacements) - f_orig[node_idx]
    
    return {
        "displacements": displacements,
        "reactions": reactions,
        "node_coords": node_coords,
        "reaction_nodes": reaction_nodes
    }
