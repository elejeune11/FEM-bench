import numpy as np
from typing import List, Dict, Callable, Optional


def FEM_1D_linear_elastic_CC0_H0_T0(
    x_min: float,
    x_max: float,
    num_elements: int,
    material_regions: List[Dict[str, float]],
    body_force_fn: Callable[[float], float],
    dirichlet_bc_list: List[Dict[str, float]],
    neumann_bc_list: Optional[List[Dict[str, float]]] = None,
    n_gauss: int = 2
) -> Dict[str, np.ndarray]:
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
    
    num_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, num_nodes)
    element_connectivity = np.array([[i, i + 1] for i in range(num_elements)], dtype=int)
    n_nodes = len(node_coords)
    
    # Assign material properties to each element
    E_vector = np.zeros(num_elements)
    A_vector = np.zeros(num_elements)
    
    for i in range(num_elements):
        x1, x2 = node_coords[element_connectivity[i]]
        x_center = 0.5 * (x1 + x2)
        assigned = False
        for region in material_regions:
            if region["coord_min"] <= x_center <= region["coord_max"]:
                E_vector[i] = region["E"]
                A_vector[i] = region["A"]
                assigned = True
                break
        if not assigned:
            raise ValueError(f"Element {i} centered at x = {x_center:.3f} is not assigned to any material region.")
    
    # Gauss quadrature
    if n_gauss == 1:
        xi_points = np.array([0.0])
        weights = np.array([2.0])
    elif n_gauss == 2:
        xi_points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        weights = np.array([1.0, 1.0])
    elif n_gauss == 3:
        xi_points = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("Only 1 to 3 Gauss points are supported.")
    
    dN_dxi = np.array([-0.5, 0.5])
    
    # Global stiffness and force vector
    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)
    
    for e in range(num_elements):
        node_ids = element_connectivity[e]
        x_elem = node_coords[node_ids]
        E, A = E_vector[e], A_vector[e]
        
        k_elem = np.zeros((2, 2))
        fe = np.zeros(2)
        
        for xi, w in zip(xi_points, weights):
            J = np.dot(dN_dxi, x_elem)
            dN_dx = dN_dxi / J
            B = dN_dx.reshape(1, 2)
            k_elem += E * A * J * w * (B.T @ B)
            
            N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
            x_phys = np.dot(N, x_elem)
            f = body_force_fn(x_phys)
            fe += w * f * J * N
        
        for a_local, a_global in enumerate(node_ids):
            F_global[a_global] += fe[a_local]
            for b_local, b_global in enumerate(node_ids):
                K_global[a_global, b_global] += k_elem[a_local, b_local]
    
    # Neumann BCs
    if neumann_bc_list is not None:
        tol = 1e-8
        for bc in neumann_bc_list:
            x_target, load = bc["x_location"], bc["load_mag"]
            matches = np.where(np.abs(node_coords - x_target) < tol)[0]
            if len(matches) != 1:
                raise ValueError(f"Invalid match at x = {x_target} (found {len(matches)} matches).")
            F_global[matches[0]] += load

    # Dirichlet BCs
    tol = 1e-8
    dirichlet_nodes = []
    u_prescribed = []
    
    for bc in dirichlet_bc_list:
        x_target = bc["x_location"]
        u_value = bc["u_prescribed"]
        matches = np.where(np.abs(node_coords - x_target) < tol)[0]
        if len(matches) != 1:
            raise ValueError(f"Invalid match at x = {x_target} (found {len(matches)} matches).")
        dirichlet_nodes.append(matches[0])
        u_prescribed.append(u_value)
    
    all_nodes = set(range(n_nodes))
    free_nodes = sorted(all_nodes - set(dirichlet_nodes))
    f = np.array(free_nodes)
    s = np.array(dirichlet_nodes)
    
    K_ff = K_global[np.ix_(f, f)]
    K_fs = K_global[np.ix_(f, s)]
    K_sf = K_global[np.ix_(s, f)]
    K_ss = K_global[np.ix_(s, s)]
    F_f = F_global[f]
    F_s = F_global[s]
    
    x_s = np.array(u_prescribed)
    x_f = np.linalg.solve(K_ff, F_f - K_fs @ x_s)
    
    x = np.zeros(n_nodes)
    x[f] = x_f
    x[s] = x_s
    
    R = K_sf @ x_f + K_ss @ x_s - F_s
    
    return {
        "displacements": x,
        "reactions": R,
        "node_coords": node_coords,
        "reaction_nodes": s
    }


def test_no_load_self_contained(fcn):
    """
    Test zero displacement and zero reaction in a fixed-free bar with no external load.

    A 2-element bar with uniform material, zero body force, and one fixed end should return:
    - Zero displacement at all nodes
    - Zero reaction at the fixed node
    - Correct output shapes and boundary condition enforcement
    """
    # Simple 2-element system: fixed at x=0, no external load
    x_min, x_max = 0.0, 2.0
    num_elements = 2

    # Uniform material properties
    E, A = 200e9, 0.01
    material_regions = [{"coord_min": 0.0, "coord_max": 2.0, "E": E, "A": A}]

    # Zero body force
    def zero_body_force(x):
        return 0.0

    # Dirichlet BC: fixed at left end
    dirichlet_bc_list = [{"x_location": 0.0, "u_prescribed": 0.0}]

    # Call solver (returns a dictionary)
    result = fcn(
        x_min, x_max, num_elements, material_regions, zero_body_force,
        dirichlet_bc_list, neumann_bc_list=None, n_gauss=2
    )

    x = result["displacements"]
    R = result["reactions"]
    reaction_nodes = result["reaction_nodes"]

    # Expected outputs
    expected_x = np.zeros(3)
    expected_R = np.zeros(1)

    np.testing.assert_array_almost_equal(x, expected_x, decimal=12)
    np.testing.assert_array_almost_equal(R, expected_R, decimal=12)

    # Additional checks
    assert x.shape == (3,)                      # 3 nodes
    assert R.shape == (1,)                      # 1 Dirichlet BC
    assert np.isclose(x[reaction_nodes[0]], 0)  # Fixed node displacement is zero


def test_analytical_solution(fcn):
    """
    Test a non-zero displacement field against a known analytical solution.
    """
    # 3-element bar under constant body force
    x_min, x_max = 0.0, 3.0
    num_elements = 3

    # Uniform material properties
    E, A = 100e9, 0.001
    material_regions = [{"coord_min": 0.0, "coord_max": 3.0, "E": E, "A": A}]
    
    # Constant body force
    f_body = 1000.0  # N/m

    def constant_body_force(x):
        return f_body
    
    # Dirichlet BC: fixed at left end
    dirichlet_bc_list = [{"x_location": 0.0, "u_prescribed": 0.0}]
    
    # Call solver
    result = fcn(
        x_min, x_max, num_elements, material_regions, constant_body_force,
        dirichlet_bc_list, neumann_bc_list=None, n_gauss=2
    )
    
    x = result["displacements"]
    R = result["reactions"]
    node_coords = result["node_coords"]
    
    # Analytical solution: u(x) = (f * x / (E * A)) * (L - x/2)
    L = x_max
    expected_displacements = (f_body * node_coords / (E * A)) * (L - 0.5 * node_coords)
    
    np.testing.assert_array_almost_equal(x, expected_displacements, decimal=10)
    
    # Monotonic displacement increase
    assert np.all(np.diff(x) > 0)
    assert x[0] == 0.0  # Fixed end displacement
    
    # Reaction force should balance total body force
    total_body_force = f_body * L
    assert abs(R[0] + total_body_force) < 1e-8


def return_all_zeros(
    x_min: float,
    x_max: float,
    num_elements: int,
    material_regions: List[Dict[str, float]],
    body_force_fn: Callable[[float], float],
    dirichlet_bc_list: List[Dict[str, float]],
    neumann_bc_list: Optional[List[Dict[str, float]]] = None,
    n_gauss: int = 2
) -> Dict[str, np.ndarray]:
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)

    # Identify Dirichlet node indices
    tol = 1e-8
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        x_target = bc["x_location"]
        matches = np.where(np.abs(node_coords - x_target) < tol)[0]
        if len(matches) != 1:
            raise ValueError(f"Invalid match at x = {x_target} (found {len(matches)} matches).")
        reaction_nodes.append(matches[0])

    x = np.zeros(n_nodes)
    R = np.zeros(len(reaction_nodes))

    return {
        "displacements": x,
        "reactions": R,
        "node_coords": node_coords,
        "reaction_nodes": np.array(reaction_nodes)
    }


def return_all_ones(
    x_min: float,
    x_max: float,
    num_elements: int,
    material_regions: List[Dict[str, float]],
    body_force_fn: Callable[[float], float],
    dirichlet_bc_list: List[Dict[str, float]],
    neumann_bc_list: Optional[List[Dict[str, float]]] = None,
    n_gauss: int = 2
) -> Dict[str, np.ndarray]:
    n_nodes = num_elements + 1
    node_coords = np.linspace(x_min, x_max, n_nodes)

    # Identify Dirichlet node indices
    tol = 1e-8
    reaction_nodes = []
    for bc in dirichlet_bc_list:
        x_target = bc["x_location"]
        matches = np.where(np.abs(node_coords - x_target) < tol)[0]
        if len(matches) != 1:
            raise ValueError(f"Invalid match at x = {x_target} (found {len(matches)} matches).")
        reaction_nodes.append(matches[0])

    x = np.ones(n_nodes)
    R = np.ones(len(reaction_nodes))

    return {
        "displacements": x,
        "reactions": R,
        "node_coords": node_coords,
        "reaction_nodes": np.array(reaction_nodes)
    }


def task_info():
    task_id = "FEM_1D_linear_elastic_CC0_H0_T0"
    task_short_description = "self contained function to solve a 1D linear elastic FEA problem"
    created_date = "2025-07-11"
    created_by = "elejeune11"
    main_fcn = FEM_1D_linear_elastic_CC0_H0_T0
    required_imports = ["import numpy as np", "from typing import Callable, List, Dict, Optional", "import pytest"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1. Fixed-free bar with no load
        [
            0.0, 1.0, 2,  # x_min, x_max, num_elements
            [{"coord_min": 0.0, "coord_max": 1.0, "E": 1e9, "A": 1.0}],  # material_regions
            lambda x: 0.0,  # body_force_fn
            [{"x_location": 0.0, "u_prescribed": 0.0}],  # dirichlet_bc_list
            None,  # neumann_bc_list
            2      # n_gauss
        ],

        # 2. Fixed-free bar with constant body force
        [
            0.0, 2.0, 4,
            [{"coord_min": 0.0, "coord_max": 2.0, "E": 2e9, "A": 0.5}],
            lambda x: 100.0,
            [{"x_location": 0.0, "u_prescribed": 0.0}],
            None,
            2
        ],

        # 3. Bar with fixed ends (Dirichlet BCs at both ends), no body force
        [
            0.0, 1.0, 2,
            [{"coord_min": 0.0, "coord_max": 1.0, "E": 1e6, "A": 1.0}],
            lambda x: 0.0,
            [
                {"x_location": 0.0, "u_prescribed": 0.0},
                {"x_location": 1.0, "u_prescribed": 0.0}
            ],
            None,
            2
        ],

        # 4. Bar with one fixed end and point load at the other
        [
            0.0, 3.0, 3,
            [{"coord_min": 0.0, "coord_max": 3.0, "E": 1e8, "A": 1.0}],
            lambda x: 0.0,
            [{"x_location": 0.0, "u_prescribed": 0.0}],
            [{"x_location": 3.0, "load_mag": -500.0}],
            2
        ],

        # 5. Bar with step in material properties
        [
            0.0, 2.0, 4,
            [
                {"coord_min": 0.0, "coord_max": 1.0, "E": 1e7, "A": 1.0},
                {"coord_min": 1.0, "coord_max": 2.0, "E": 1e9, "A": 0.5}
            ],
            lambda x: 50.0,
            [{"x_location": 0.0, "u_prescribed": 0.0}],
            None,
            3
        ]
    ]
    test_cases = [{"test_code": test_no_load_self_contained, "expected_failures": [return_all_ones]}, {"test_code": test_analytical_solution, "expected_failures": [return_all_zeros]}]
    return {
        "task_id": task_id,
        "task_short_description": task_short_description,
        "created_date": created_date,
        "created_by": created_by,
        "main_fcn": main_fcn,
        "required_imports": required_imports,
        "fcn_dependencies": fcn_dependencies,
        "reference_verification_inputs": reference_verification_inputs,
        "test_cases": test_cases,
        # "python_version": "version_number",
        # "package_versions": {"numpy": "version_number", },
    }