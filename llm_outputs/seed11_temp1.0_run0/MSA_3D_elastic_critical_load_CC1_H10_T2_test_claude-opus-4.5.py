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
            (lambda_cr, mode_shape) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_numerical = lambda_cr * P_ref
            rel_error = abs(P_cr_numerical - P_euler) / P_euler
            assert rel_error < 1e-05, f'Failed for r={r}, L={L}: P_cr_numerical={P_cr_numerical}, P_euler={P_euler}, rel_error={rel_error}'
            assert lambda_cr > 0, f'Critical load factor should be positive, got {lambda_cr}'
            assert mode_shape.shape == (6 * n_nodes,), f'Mode shape has wrong size'

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
    b = 1.0
    h = 2.0
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = b * h ** 3 * (1 / 3 - 0.21 * (h / b) * (1 - h ** 4 / (12 * b ** 4)))
    n_nodes = n_elements + 1
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 0] = np.linspace(0, L, n_nodes)
    elements_base = []
    for i in range(n_elements):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
        elements_base.append(elem)
    boundary_conditions_base = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1.0
    load_dir_base = np.array([-1.0, 0.0, 0.0])
    nodal_loads_base = {n_nodes - 1: [load_dir_base[0] * P_ref, load_dir_base[1] * P_ref, load_dir_base[2] * P_ref, 0.0, 0.0, 0.0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions_base, nodal_loads_base)
    theta_z = np.pi / 4
    theta_x = np.pi / 6
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    R = Rx @ Rz
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ np.array([0.0, 0.0, 1.0])
    elements_rot = []
    for i in range(n_elements):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}
        elements_rot.append(elem)
    boundary_conditions_rot = {0: [1, 1, 1, 1, 1, 1]}
    load_dir_rot = R @ load_dir_base
    nodal_loads_rot = {n_nodes - 1: [load_dir_rot[0] * P_ref, load_dir_rot[1] * P_ref, load_dir_rot[2] * P_ref, 0.0, 0.0, 0.0]}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions_rot, nodal_loads_rot)
    rel_error_lambda = abs(lambda_rot - lambda_base) / abs(lambda_base)
    assert rel_error_lambda < 1e-10, f'Critical load factors differ: lambda_base={lambda_base}, lambda_rot={lambda_rot}, rel_error={rel_error_lambda}'
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    scale_rot = np.max(np.abs(mode_rot))
    scale_base_t = np.max(np.abs(mode_base_transformed))
    if scale_rot > 1e-12 and scale_base_t > 1e-12:
        mode_rot_norm = mode_rot / scale_rot
        mode_base_t_norm = mode_base_transformed / scale_base_t
        dot_product = np.dot(mode_rot_norm, mode_base_t_norm)
        if dot_product < 0:
            mode_base_t_norm = -mode_base_t_norm
        mode_diff = np.linalg.norm(mode_rot_norm - mode_base_t_norm)
        assert mode_diff < 1e-06, f'Mode shapes differ after transformation: norm diff = {mode_diff}'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000.0
    nu = 0.3
    L = 30.0
    r = 0.8
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    n_elements_list = [2, 4, 8, 16, 32, 64]
    errors = []
    P_ref = 1.0
    for n_elements in n_elements_list:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_elements):
            elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])}
            elements.append(elem)
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_cr, mode_shape) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_numerical = lambda_cr * P_ref
        rel_error = abs(P_cr_numerical - P_euler) / P_euler
        errors.append(rel_error)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] * 1.1, f'Convergence not monotonic: error[{n_elements_list[i]}]={errors[i]} > error[{n_elements_list[i - 1]}]={errors[i - 1]}'
    assert errors[-1] < 1e-06, f'Finest mesh (n={n_elements_list[-1]}) did not achieve sufficient accuracy: rel_error={errors[-1]}'
    for i in range(2, len(errors) - 1):
        if errors[i] > 1e-12:
            ratio = errors[i - 1] / errors[i]
            assert ratio > 3.0, f'Convergence rate too slow between n={n_elements_list[i - 1]} and n={n_elements_list[i]}: ratio={ratio}, expected ~4'