def test_euler_buckling_cantilever_circular_param_sweep(fcn: Callable):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    E = 210000.0
    nu = 0.3
    n_elements = 10
    P_ref = 1.0
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = np.pi * r ** 4 / 2
            euler_critical = np.pi ** 2 * E * I / (4 * L ** 2)
            node_coords = np.array([[0.0, 0.0, z] for z in np.linspace(0, L, n_elements + 1)])
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_elements: [0.0, P_ref, 0.0, 0.0, 0.0, 0.0]}
            (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            predicted_critical = lambda_crit * P_ref
            relative_error = abs(predicted_critical - euler_critical) / euler_critical
            assert lambda_crit > 0, f'Critical load factor must be positive for r={r}, L={L}'
            assert relative_error < 0.01, f'Relative error {relative_error} exceeds 1% for r={r}, L={L}'
            assert len(mode) == 6 * len(node_coords), f'Mode shape length mismatch for r={r}, L={L}'
            assert np.allclose(mode[0:6], 0.0, atol=1e-10), f'Fixed end DOFs not zero for r={r}, L={L}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn: Callable):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    L = 20.0
    E = 210000.0
    nu = 0.3
    n_elements = 8
    P_ref = 1.0
    b = 0.5
    h = 1.0
    A = b * h
    I_y = b * h ** 3 / 12
    I_z = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    node_coords_base = np.array([[0.0, 0.0, z] for z in np.linspace(0, L, n_elements + 1)])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    boundary_conditions_base = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_elements: [0.0, P_ref, 0.0, 0.0, 0.0, 0.0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions_base, nodal_loads_base)
    angle = np.pi / 6
    (c, s) = (np.cos(angle), np.sin(angle))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = node_coords_base @ R.T
    elements_rot = []
    for i in range(n_elements):
        local_z_rot = R @ np.array([0.0, 1.0, 0.0])
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    load_rot = R @ np.array([0.0, P_ref, 0.0])
    nodal_loads_rot = {n_elements: [load_rot[0], load_rot[1], load_rot[2], 0.0, 0.0, 0.0]}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions_base, nodal_loads_rot)
    assert np.isclose(lambda_base, lambda_rot, rtol=0.0001), f'Critical load factors differ: base={lambda_base}, rotated={lambda_rot}'
    n_nodes = len(node_coords_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for node_idx in range(n_nodes):
        for dof_type in range(2):
            row_start = node_idx * 6 + dof_type * 3
            col_start = node_idx * 6 + dof_type * 3
            T[row_start:row_start + 3, col_start:col_start + 3] = R
    mode_base_normalized = mode_base / (np.max(np.abs(mode_base)) + 1e-14)
    mode_rot_normalized = mode_rot / (np.max(np.abs(mode_rot)) + 1e-14)
    mode_transformed = T @ mode_base_normalized
    mode_transformed_normalized = mode_transformed / (np.max(np.abs(mode_transformed)) + 1e-14)
    correlation = np.abs(np.dot(mode_rot_normalized, mode_transformed_normalized)) / (np.linalg.norm(mode_rot_normalized) * np.linalg.norm(mode_transformed_normalized) + 1e-14)
    assert correlation > 0.95, f'Mode shape correlation {correlation} is too low; expected > 0.95'

def test_cantilever_euler_buckling_mesh_convergence(fcn: Callable):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 25.0
    E = 210000.0
    nu = 0.3
    r = 0.75
    P_ref = 1.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    euler_critical = np.pi ** 2 * E * I / (4 * L ** 2)
    n_element_list = [4, 8, 16, 32]
    relative_errors = []
    for n_elements in n_element_list:
        node_coords = np.array([[0.0, 0.0, z] for z in np.linspace(0, L, n_elements + 1)])
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [0.0, P_ref, 0.0, 0.0, 0.0, 0.0]}
        (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        predicted_critical = lambda_crit * P_ref
        relative_error = abs(predicted_critical - euler_critical) / euler_critical
        relative_errors.append(relative_error)
        assert lambda_crit > 0, f'Critical load factor must be positive for n_elements={n_elements}'
        assert len(mode) == 6 * len(node_coords), f'Mode shape length mismatch for n_elements={n_elements}'
    assert relative_errors[-1] < 0.0001, f'Finest mesh (n_elements={n_element_list[-1]}) relative error {relative_errors[-1]} exceeds 1e-4'
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i] > relative_errors[i + 1], f'Relative error did not decrease from mesh {n_element_list[i]} to {n_element_list[i + 1]}: {relative_errors[i]} vs {relative_errors[i + 1]}'