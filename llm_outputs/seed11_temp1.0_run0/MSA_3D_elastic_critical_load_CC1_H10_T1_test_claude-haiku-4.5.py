def test_euler_buckling_cantilever_circular_param_sweep(fcn: Callable):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    n_elements = 10
    tolerance = 0.01
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I_y = np.pi * r ** 4 / 4
            I_z = np.pi * r ** 4 / 4
            J = np.pi * r ** 4 / 2
            z_coords = np.linspace(0, L, n_elements + 1)
            node_coords = np.array([[0, 0, z] for z in z_coords])
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 1, 0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref = 1.0
            nodal_loads = {n_elements: [-P_ref, 0, 0, 0, 0, 0]}
            (lambda_crit, mode_shape) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_euler = np.pi ** 2 * E * I_z / (4 * L ** 2)
            P_numerical = lambda_crit * P_ref
            relative_error = abs(P_numerical - P_euler) / P_euler
            assert relative_error < tolerance, f'r={r}, L={L}: relative error {relative_error:.6f} exceeds tolerance {tolerance}'
            assert np.linalg.norm(mode_shape) > 0, f'r={r}, L={L}: mode shape is zero'
            assert lambda_crit > 0, f'r={r}, L={L}: critical load factor is not positive'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn: Callable):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    b = 0.5
    h = 1.0
    A = b * h
    I_y = b * h ** 3 / 12
    I_z = h * b ** 3 / 12
    J = min(b, h) ** 3 * max(b, h) / 3
    n_elements = 8
    z_coords_base = np.linspace(0, L, n_elements + 1)
    node_coords_base = np.array([[0, 0, z] for z in z_coords_base])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 1, 0])})
    boundary_conditions_base = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_elements: [-1.0, 0, 0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions_base, nodal_loads_base)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    node_coords_rot = node_coords_base @ R.T
    elements_rot = []
    for elem in elements_base:
        local_z_rot = R @ np.array(elem['local_z'])
        elements_rot.append({'node_i': elem['node_i'], 'node_j': elem['node_j'], 'E': elem['E'], 'nu': elem['nu'], 'A': elem['A'], 'I_y': elem['I_y'], 'I_z': elem['I_z'], 'J': elem['J'], 'local_z': local_z_rot})
    nodal_load_base = np.array([-1.0, 0, 0, 0, 0, 0])
    load_rotated_spatial = np.concatenate([R @ nodal_load_base[:3], R @ nodal_load_base[3:]])
    nodal_loads_rot = {n_elements: load_rotated_spatial}
    boundary_conditions_rot = {0: [1, 1, 1, 1, 1, 1]}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions_rot, nodal_loads_rot)
    assert abs(lambda_base - lambda_rot) / max(abs(lambda_base), 1e-10) < 1e-06, f'Critical load factors do not match: base={lambda_base}, rot={lambda_rot}'
    n_nodes = len(node_coords_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    if np.linalg.norm(mode_rot) > 1e-10 and np.linalg.norm(mode_base_transformed) > 1e-10:
        scale_factor = np.dot(mode_rot, mode_base_transformed) / (np.linalg.norm(mode_base_transformed) ** 2 + 1e-16)
        mode_base_scaled = scale_factor * mode_base_transformed
        relative_error = np.linalg.norm(mode_rot - mode_base_scaled) / (np.linalg.norm(mode_rot) + 1e-10)
        assert relative_error < 0.0001, f'Rotated mode does not match transformed base mode; relative error={relative_error:.6e}'

def test_cantilever_euler_buckling_mesh_convergence(fcn: Callable):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    r = 0.75
    A = np.pi * r ** 2
    I_z = np.pi * r ** 4 / 4
    I_y = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    P_euler = np.pi ** 2 * E * I_z / (4 * L ** 2)
    n_element_list = [5, 10, 20, 40]
    relative_errors = []
    for n_elements in n_element_list:
        z_coords = np.linspace(0, L, n_elements + 1)
        node_coords = np.array([[0, 0, z] for z in z_coords])
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 1, 0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [-1.0, 0, 0, 0, 0, 0]}
        (lambda_crit, mode_shape) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_numerical = lambda_crit * 1.0
        relative_error = abs(P_numerical - P_euler) / P_euler
        relative_errors.append(relative_error)
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i] >= relative_errors[i + 1] * 0.95, f'Errors not monotonically decreasing: {relative_errors}'
    finest_error = relative_errors[-1]
    assert finest_error < 0.001, f'Finest mesh (n={n_element_list[-1]}) achieves only {finest_error:.6f} relative error; expected < 1e-3'
    for lambda_val in relative_errors:
        assert P_euler * (1 - lambda_val) > 0, 'Numerical load became negative'