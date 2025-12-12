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
    P_ref = 1000.0
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = np.pi * r ** 4 / 2
            analytical_lambda = np.pi ** 2 * E * I / (4 * L ** 2 * P_ref)
            node_coords = np.zeros((n_elements + 1, 3))
            node_coords[:, 2] = np.linspace(0, L, n_elements + 1)
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_elements: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lambda_crit > 0, f'Critical load factor must be positive for r={r}, L={L}'
            numerical_load = lambda_crit * P_ref
            relative_error = abs(numerical_load - analytical_lambda * P_ref) / (analytical_lambda * P_ref)
            assert relative_error < 0.01, f'Relative error {relative_error:.6f} exceeds 1% for r={r}, L={L}. Analytical: {analytical_lambda * P_ref:.4f}, Numerical: {numerical_load:.4f}'
            assert len(mode) == 6 * (n_elements + 1), f'Mode shape vector length mismatch for r={r}, L={L}'
            assert np.allclose(mode[0:6], 0.0, atol=1e-10), f'Fixed boundary condition DOFs should be zero for r={r}, L={L}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn: Callable):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 210000.0
    nu = 0.3
    L = 20.0
    n_elements = 8
    P_ref = 500.0
    b = 0.5
    h = 1.0
    A = b * h
    I_y = b * h ** 3 / 12
    I_z = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    node_coords_base = np.zeros((n_elements + 1, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_elements + 1)
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    boundary_conditions_base = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_elements: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions_base, nodal_loads_base)
    angle = np.pi / 6
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = node_coords_base @ R.T
    elements_rot = []
    for elem in elements_base:
        local_z_rot = R @ elem['local_z']
        elements_rot.append({'node_i': elem['node_i'], 'node_j': elem['node_j'], 'E': elem['E'], 'nu': elem['nu'], 'A': elem['A'], 'I_y': elem['I_y'], 'I_z': elem['I_z'], 'J': elem['J'], 'local_z': local_z_rot})
    boundary_conditions_rot = {0: [1, 1, 1, 1, 1, 1]}
    load_rot = R @ np.array([0.0, 0.0, -P_ref])
    nodal_loads_rot = {n_elements: np.concatenate([load_rot, [0.0, 0.0, 0.0]])}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions_rot, nodal_loads_rot)
    assert np.isclose(lambda_base, lambda_rot, rtol=1e-06), f'Critical load factors differ: base={lambda_base:.8f}, rotated={lambda_rot:.8f}'
    n_nodes = n_elements + 1
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    scale_factor = np.dot(mode_rot, mode_base_transformed) / np.dot(mode_base_transformed, mode_base_transformed)
    mode_base_scaled = scale_factor * mode_base_transformed
    correlation = np.linalg.norm(mode_rot - mode_base_scaled) / np.linalg.norm(mode_rot)
    assert correlation < 0.0001, f'Rotated mode does not match transformed base mode. Correlation error: {correlation:.8f}'

def test_cantilever_euler_buckling_mesh_convergence(fcn: Callable):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000.0
    nu = 0.3
    L = 25.0
    r = 0.75
    P_ref = 1000.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    analytical_lambda = np.pi ** 2 * E * I / (4 * L ** 2 * P_ref)
    n_element_list = [4, 8, 16, 32]
    relative_errors = []
    for n_elements in n_element_list:
        node_coords = np.zeros((n_elements + 1, 3))
        node_coords[:, 2] = np.linspace(0, L, n_elements + 1)
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lambda_crit > 0, f'Critical load factor must be positive for n_elements={n_elements}'
        numerical_load = lambda_crit * P_ref
        relative_error = abs(numerical_load - analytical_lambda * P_ref) / (analytical_lambda * P_ref)
        relative_errors.append(relative_error)
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i] > relative_errors[i + 1], f'Relative error should decrease with mesh refinement: n_elem={n_element_list[i]} error={relative_errors[i]:.8f}, n_elem={n_element_list[i + 1]} error={relative_errors[i + 1]:.8f}'
    assert relative_errors[-1] < 0.0001, f'Finest mesh (n_elements={n_element_list[-1]}) should achieve high accuracy. Relative error: {relative_errors[-1]:.8f}'