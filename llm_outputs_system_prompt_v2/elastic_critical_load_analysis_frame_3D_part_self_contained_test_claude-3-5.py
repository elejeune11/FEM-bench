def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Verify critical loads match Euler theory across parameter sweep of circular cantilevers"""
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elements = 10
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4
        J = 2 * I
        I_rho = 2 * I
        for L in lengths:
            x = np.linspace(0, L, n_elements + 1)
            nodes = np.column_stack((np.zeros_like(x), np.zeros_like(x), x))
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
            bcs = {0: [True] * 6}
            P = 1.0
            loads = {n_elements: [0, P, 0, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_numerical = lambda_cr * P
            assert_allclose(P_numerical, P_euler, rtol=0.01)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Verify critical load and mode shape invariance under rigid rotation"""
    E = 200000000000.0
    nu = 0.3
    h = 0.1
    b = 0.05
    L = 5.0
    n_elements = 10
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = min(Iy, Iz) / 3
    I_rho = Iy + Iz
    x = np.linspace(0, L, n_elements + 1)
    nodes_base = np.column_stack((np.zeros_like(x), np.zeros_like(x), x))
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    loads = {n_elements: [0, 1.0, 0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads)
    theta = np.pi / 4
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    nodes_rot = nodes_base @ R.T
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': R @ [0, 1, 0]})
    loads_rot = {n_elements: np.concatenate((R @ [0, 1.0, 0], [0, 0, 0]))}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    assert_allclose(lambda_rot, lambda_base, rtol=1e-10)
    T = block_diag(*[R] * 2)
    T_full = block_diag(*[T] * (n_elements + 1))
    mode_rot_normalized = mode_rot / np.linalg.norm(mode_rot)
    mode_base_transformed = T_full @ mode_base
    mode_base_transformed_normalized = mode_base_transformed / np.linalg.norm(mode_base_transformed)
    alignment = np.abs(np.dot(mode_rot_normalized, mode_base_transformed_normalized))
    assert_allclose(alignment, 1.0, rtol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify convergence to Euler buckling load with mesh refinement"""
    E = 200000000000.0
    nu = 0.3
    r = 0.05
    L = 10.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    relative_errors = []
    for n_elements in n_elements_list:
        x = np.linspace(0, L, n_elements + 1)
        nodes = np.column_stack((np.zeros_like(x), np.zeros_like(x), x))
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
        bcs = {0: [True] * 6}
        loads = {n_elements: [0, 1.0, 0, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_numerical = lambda_cr
        relative_errors.append(abs(P_numerical - P_euler) / P_euler)
    convergence_rates = np.diff(np.log(relative_errors)) / np.diff(np.log(n_elements_list))
    assert np.all(convergence_rates < -1.0)
    assert relative_errors[-1] < 0.0001