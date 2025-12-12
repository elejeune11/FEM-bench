def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10^-5.
    """
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    E = 210000.0
    nu = 0.3
    n_elements = 10
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = np.pi * r ** 4 / 2
            P_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
            node_coords = np.zeros((n_elements + 1, 3))
            for i in range(n_elements + 1):
                node_coords[i, 2] = L * i / n_elements
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref = 1000.0
            nodal_loads = {n_elements: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lambda_crit > 0.0, f'Critical load factor must be positive for r={r}, L={L}'
            P_predicted = lambda_crit * P_ref
            rel_error = abs(P_predicted - P_analytical) / P_analytical
            assert rel_error < 0.1, f'Relative error {rel_error:.4f} exceeds 10% for r={r}, L={L}. Analytical: {P_analytical:.2f}, Predicted: {P_predicted:.2f}'
            assert np.max(np.abs(mode)) > 1e-10, f'Mode vector is too small for r={r}, L={L}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
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
    b = 0.1
    h = 0.2
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = 0.1 * b * h * (b ** 2 + h ** 2) / 12.0
    node_coords_base = np.zeros((n_elements + 1, 3))
    for i in range(n_elements + 1):
        node_coords_base[i, 2] = L * i / n_elements
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    boundary_conditions_base = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1000.0
    nodal_loads_base = {n_elements: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions_base, nodal_loads_base)
    angle = np.pi / 2.0
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]])
    node_coords_rotated = node_coords_base @ R.T
    elements_rotated = []
    for elem in elements_base:
        elem_rot = elem.copy()
        if 'local_z' in elem_rot:
            elem_rot['local_z'] = R @ np.array(elem_rot['local_z'])
        elements_rotated.append(elem_rot)
    load_rot = R @ np.array([0.0, 0.0, -P_ref])
    nodal_loads_rotated = {n_elements: load_rot}
    (lambda_rotated, mode_rotated) = fcn(node_coords_rotated, elements_rotated, boundary_conditions_base, nodal_loads_rotated)
    assert np.abs(lambda_rotated - lambda_base) / lambda_base < 1e-06, f'Eigenvalue changed under rotation: {lambda_base:.8f} vs {lambda_rotated:.8f}'
    n_nodes = n_elements + 1
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    norm_base = np.linalg.norm(mode_base_transformed)
    norm_rotated = np.linalg.norm(mode_rotated)
    if norm_base > 1e-12 and norm_rotated > 1e-12:
        mode_base_transformed_normalized = mode_base_transformed / norm_base
        mode_rotated_normalized = mode_rotated / norm_rotated
        cos_angle = np.abs(np.dot(mode_base_transformed_normalized, mode_rotated_normalized))
        assert cos_angle > 0.99, f'Rotated mode does not match transformed base mode. Cosine: {cos_angle:.6f}'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
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
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    P_analytical = np.pi ** 2 * E * I / (4.0 * L ** 2)
    n_elements_list = [4, 8, 16, 32]
    relative_errors = []
    for n_elements in n_elements_list:
        node_coords = np.zeros((n_elements + 1, 3))
        for i in range(n_elements + 1):
            node_coords[i, 2] = L * i / n_elements
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        P_ref = 1000.0
        nodal_loads = {n_elements: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_crit, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_predicted = lambda_crit * P_ref
        rel_error = abs(P_predicted - P_analytical) / P_analytical
        relative_errors.append(rel_error)
        assert rel_error > 0.0, f'Relative error must be positive for n_elements={n_elements}'
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i] > relative_errors[i + 1], f'Convergence not monotonic: error at {n_elements_list[i]} elements ({relative_errors[i]:.6f}) not > error at {n_elements_list[i + 1]} elements ({relative_errors[i + 1]:.6f})'
    finest_error = relative_errors[-1]
    assert finest_error < 0.01, f'Finest mesh (n_elements={n_elements_list[-1]}) achieves only {finest_error * 100:.2f}% error; should be < 1%'