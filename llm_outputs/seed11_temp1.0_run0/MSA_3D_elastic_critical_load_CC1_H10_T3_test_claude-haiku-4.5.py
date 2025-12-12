def test_euler_buckling_cantilever_circular_param_sweep(fcn: Callable):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    E = 200000.0
    nu = 0.3
    n_elements = 10
    tolerance = 0.001
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
            n_nodes = n_elements + 1
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0.0, 1.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref = 1.0
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lambda_crit, mode_shape) = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
            P_cr_numerical = lambda_crit * P_ref
            relative_error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
            assert relative_error < tolerance, f'For r={r}, L={L}: relative error {relative_error:.6e} exceeds tolerance {tolerance}. Analytical: {P_cr_analytical:.6e}, Numerical: {P_cr_numerical:.6e}'
            assert np.linalg.norm(mode_shape) > 0, f'Mode shape is zero for r={r}, L={L}'
            assert lambda_crit > 0, f'Eigenvalue not positive for r={r}, L={L}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn: Callable):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 200000.0
    nu = 0.3
    b = 0.1
    h = 0.2
    A = b * h
    I_y = b * h ** 3 / 12
    I_z = h * b ** 3 / 12
    J = b * h / 12 * (b ** 2 + h ** 2)
    L = 10.0
    n_elements = 8
    P_ref = 1.0
    n_nodes = n_elements + 1
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_nodes)
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    boundary_conditions_base = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lambda_base, mode_base) = fcn(node_coords=node_coords_base, elements=elements_base, boundary_conditions=boundary_conditions_base, nodal_loads=nodal_loads_base)
    theta = np.pi / 4
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
    node_coords_rot = node_coords_base @ R.T
    local_z_base = np.array([0.0, 1.0, 0.0])
    local_z_rot = local_z_base @ R.T
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    load_vec_base = np.array([0.0, 0.0, -P_ref])
    load_vec_rot = load_vec_base @ R.T
    boundary_conditions_rot = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_rot = {n_nodes - 1: load_vec_rot.tolist() + [0.0, 0.0, 0.0]}
    (lambda_rot, mode_rot) = fcn(node_coords=node_coords_rot, elements=elements_rot, boundary_conditions=boundary_conditions_rot, nodal_loads=nodal_loads_rot)
    assert abs(lambda_base - lambda_rot) / (abs(lambda_base) + 1e-12) < 0.0001, f'Eigenvalues differ: base={lambda_base:.8e}, rotated={lambda_rot:.8e}'
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for node_id in range(n_nodes):
        dof_start = 6 * node_id
        T[dof_start:dof_start + 3, dof_start:dof_start + 3] = R
        T[dof_start + 3:dof_start + 6, dof_start + 3:dof_start + 6] = R
    free_dofs = [i for i in range(6 * n_nodes) if i >= 6]
    mode_base_free = mode_base[free_dofs]
    mode_rot_free = mode_rot[free_dofs]
    T_free = T[np.ix_(free_dofs, free_dofs)]
    mode_base_rotated = T_free @ mode_base_free
    norm_base = np.linalg.norm(mode_base_rotated)
    norm_rot = np.linalg.norm(mode_rot_free)
    if norm_base > 1e-12 and norm_rot > 1e-12:
        mode_base_rotated_normalized = mode_base_rotated / norm_base
        mode_rot_normalized = mode_rot_free / norm_rot
        correlation_same = np.abs(np.dot(mode_base_rotated_normalized, mode_rot_normalized))
        correlation_opp = np.abs(np.dot(-mode_base_rotated_normalized, mode_rot_normalized))
        correlation = max(correlation_same, correlation_opp)
        assert correlation > 0.99, f'Rotated mode shape does not match transformed base mode. Correlation: {correlation:.6e}'

def test_cantilever_euler_buckling_mesh_convergence(fcn: Callable):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 200000.0
    nu = 0.3
    r = 1.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    L = 20.0
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    n_element_sequence = [5, 10, 20, 40]
    relative_errors = []
    for n_elements in n_element_sequence:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        P_ref = 1.0
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_crit, mode_shape) = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
        P_cr_numerical = lambda_crit * P_ref
        relative_error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
        relative_errors.append(relative_error)
        assert lambda_crit > 0, f'Eigenvalue not positive for n_elements={n_elements}'
        assert np.linalg.norm(mode_shape) > 0, f'Mode shape is zero for n_elements={n_elements}'
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i + 1] <= relative_errors[i] * 1.01, f'Mesh refinement did not improve accuracy. n_elements={n_element_sequence[i]}: error={relative_errors[i]:.6e}, n_elements={n_element_sequence[i + 1]}: error={relative_errors[i + 1]:.6e}'
    finest_error = relative_errors[-1]
    assert finest_error < 0.001, f'Finest mesh (n_elements={n_element_sequence[-1]}) relative error {finest_error:.6e} does not meet accuracy requirement 1e-3'