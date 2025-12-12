def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 210000.0
    nu = 0.3
    n_elements = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4.0
            J = np.pi * r ** 4 / 2.0
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            n_nodes = n_elements + 1
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_elements):
                elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])}
                elements.append(elem)
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref = 1.0
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lambda_cr, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_numerical = lambda_cr * P_ref
            rel_error = abs(P_cr_numerical - P_euler) / P_euler
            assert rel_error < 0.01, f'Euler buckling mismatch for r={r}, L={L}: numerical={P_cr_numerical:.6f}, analytical={P_euler:.6f}, rel_error={rel_error:.6e}'
            assert mode.shape == (6 * n_nodes,), f'Mode shape has wrong size for r={r}, L={L}'
            assert lambda_cr > 0, f'Critical load factor should be positive for r={r}, L={L}'

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
    L = 15.0
    n_elements = 8
    b = 1.0
    h = 2.0
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = (b * h ** 3 + h * b ** 3) / 12.0
    n_nodes = n_elements + 1
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 0] = np.linspace(0, L, n_nodes)
    elements_base = []
    for i in range(n_elements):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
        elements_base.append(elem)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1.0
    nodal_loads_base = {n_nodes - 1: [-P_ref, 0.0, 0.0, 0.0, 0.0, 0.0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    theta = np.pi / 4.0
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_base = np.array([0.0, 0.0, 1.0])
    local_z_rot = R @ local_z_base
    elements_rot = []
    for i in range(n_elements):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}
        elements_rot.append(elem)
    load_base = np.array([-P_ref, 0.0, 0.0])
    load_rot = R @ load_base
    nodal_loads_rot = {n_nodes - 1: [load_rot[0], load_rot[1], load_rot[2], 0.0, 0.0, 0.0]}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_error_lambda = abs(lambda_rot - lambda_base) / abs(lambda_base)
    assert rel_error_lambda < 1e-06, f'Critical load factor should be orientation-invariant: base={lambda_base:.8f}, rotated={lambda_rot:.8f}, rel_error={rel_error_lambda:.2e}'
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    norm_rot = np.linalg.norm(mode_rot)
    norm_trans = np.linalg.norm(mode_base_transformed)
    if norm_rot > 1e-12 and norm_trans > 1e-12:
        mode_rot_normalized = mode_rot / norm_rot
        mode_trans_normalized = mode_base_transformed / norm_trans
        dot_product = abs(np.dot(mode_rot_normalized, mode_trans_normalized))
        assert dot_product > 0.99, f'Rotated mode should match transformed base mode: dot_product={dot_product:.6f}'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000.0
    nu = 0.3
    L = 20.0
    r = 0.8
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    element_counts = [2, 4, 8, 16, 32]
    errors = []
    for n_elements in element_counts:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_elements):
            elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])}
            elements.append(elem)
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        P_ref = 1.0
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_cr, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_numerical = lambda_cr * P_ref
        rel_error = abs(P_cr_numerical - P_euler) / P_euler
        errors.append(rel_error)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] * 1.1, f'Error should decrease with mesh refinement: n_elem={element_counts[i - 1]} error={errors[i - 1]:.2e}, n_elem={element_counts[i]} error={errors[i]:.2e}'
    assert errors[-1] < 0.001, f'Finest mesh (n_elem={element_counts[-1]}) should achieve < 0.1% error, got {errors[-1] * 100:.4f}%'
    if len(errors) >= 3:
        ratio_1 = errors[-3] / errors[-2] if errors[-2] > 1e-15 else float('inf')
        ratio_2 = errors[-2] / errors[-1] if errors[-1] > 1e-15 else float('inf')
        assert ratio_1 > 2.0, f'Convergence rate too slow: ratio={ratio_1:.2f}'
        assert ratio_2 > 2.0, f'Convergence rate too slow: ratio={ratio_2:.2f}'