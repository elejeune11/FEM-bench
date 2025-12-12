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
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = np.pi * r ** 4 / 2
            node_coords = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0, 1, 0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref = 1.0
            nodal_loads = {n_elements: [0, 0, -P_ref, 0, 0, 0]}
            (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_crit_numerical = lambda_crit * P_ref
            P_crit_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
            relative_error = abs(P_crit_numerical - P_crit_analytical) / P_crit_analytical
            assert relative_error < 0.01, f'r={r}, L={L}: relative error {relative_error} exceeds 1e-2'
            assert lambda_crit > 0, f'r={r}, L={L}: critical load factor must be positive'
            assert len(mode) == 6 * (n_elements + 1), f'r={r}, L={L}: mode shape has incorrect length'

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
    (b, h) = (0.5, 1.0)
    A = b * h
    I_y = b * h ** 3 / 12
    I_z = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    n_elements = 8
    node_coords_base = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 1, 0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1.0
    nodal_loads = {n_elements: [0, 0, -P_ref, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads)
    theta = np.pi / 6
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    node_coords_rot = node_coords_base @ R.T
    elements_rot = []
    for i in range(n_elements):
        local_z_rot = R @ np.array([0, 1, 0])
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    load_rot = R @ np.array([0, 0, -P_ref])
    nodal_loads_rot = {n_elements: [load_rot[0], load_rot[1], load_rot[2], 0, 0, 0]}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert abs(lambda_base - lambda_rot) / lambda_base < 1e-06, f'Critical load factors differ: {lambda_base} vs {lambda_rot}'
    n_nodes = n_elements + 1
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for node_idx in range(n_nodes):
        row_start = 6 * node_idx
        T[row_start:row_start + 3, row_start:row_start + 3] = R
        T[row_start + 3:row_start + 6, row_start + 3:row_start + 6] = R
    mode_base_transformed = T @ mode_base
    scale_factor = np.dot(mode_rot, mode_base_transformed) / np.dot(mode_base_transformed, mode_base_transformed)
    mode_base_scaled = scale_factor * mode_base_transformed
    relative_error = np.linalg.norm(mode_rot - mode_base_scaled) / np.linalg.norm(mode_rot)
    assert relative_error < 1e-05, f'Mode shapes differ after rotation: relative error {relative_error}'

def test_cantilever_euler_buckling_mesh_convergence(fcn: Callable):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 25.0
    r = 0.75
    E = 210000.0
    nu = 0.3
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    P_crit_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    n_element_list = [5, 10, 20, 40]
    relative_errors = []
    for n_elements in n_element_list:
        node_coords = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0, 1, 0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        P_ref = 1.0
        nodal_loads = {n_elements: [0, 0, -P_ref, 0, 0, 0]}
        (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_crit_numerical = lambda_crit * P_ref
        relative_error = abs(P_crit_numerical - P_crit_analytical) / P_crit_analytical
        relative_errors.append(relative_error)
        assert lambda_crit > 0, f'n_elements={n_elements}: critical load factor must be positive'
        assert len(mode) == 6 * (n_elements + 1), f'n_elements={n_elements}: mode shape has incorrect length'
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i] > relative_errors[i + 1], f'Error not decreasing: {relative_errors[i]} vs {relative_errors[i + 1]}'
    assert relative_errors[-1] < 0.0001, f'Finest mesh (n_elements={n_element_list[-1]}) relative error {relative_errors[-1]} exceeds 1e-4'