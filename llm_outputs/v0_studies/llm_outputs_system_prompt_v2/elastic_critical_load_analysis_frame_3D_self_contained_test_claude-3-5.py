def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Verify critical loads match Euler theory across parameter sweep of circular cantilevers"""
    E = 200000000000.0
    nu = 0.3
    for r in [0.5, 0.75, 1.0]:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4
        J = 2 * I
        I_rho = 2 * I
        for L in [10.0, 20.0, 40.0]:
            n_elem = 10
            z = np.linspace(0, L, n_elem + 1)
            nodes = np.zeros((n_elem + 1, 3))
            nodes[:, 2] = z
            elements = []
            for i in range(n_elem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': None})
            bcs = {0: [True] * 6}
            P = 1000.0
            loads = {n_elem: [0, 0, -P, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_cr_computed = lambda_cr * P
            P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            assert_allclose(P_cr_computed, P_cr_euler, rtol=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Verify critical load and mode shape invariance under rigid rotation"""
    E = 200000000000.0
    nu = 0.3
    h = 0.1
    b = 0.05
    L = 5.0
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    I_rho = Iy + Iz
    n_elem = 10
    nodes_base = np.zeros((n_elem + 1, 3))
    nodes_base[:, 2] = np.linspace(0, L, n_elem + 1)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    nodes_rot = nodes_base @ R.T
    elements_base = []
    elements_rot = []
    for i in range(n_elem):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': None})
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': R @ np.array([0, 0, 1])})
    bcs = {0: [True] * 6}
    P = 1000.0
    loads_base = {n_elem: [0, 0, -P, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    P_rot = R @ np.array([0, 0, -P])
    loads_rot = {n_elem: [*P_rot, 0, 0, 0]}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    T = block_diag(*[R] * 2)
    T_full = block_diag(*[T] * (n_elem + 1))
    assert_allclose(lambda_base, lambda_rot, rtol=1e-10)
    scale = np.linalg.norm(mode_rot) / np.linalg.norm(T_full @ mode_base)
    assert_allclose(mode_rot, scale * T_full @ mode_base, rtol=1e-10, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify convergence to analytical solution with mesh refinement"""
    E = 200000000000.0
    nu = 0.3
    r = 0.05
    L = 10.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P = 1000.0
    P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elem_list = [4, 8, 16, 32, 64]
    errors = []
    for n_elem in n_elem_list:
        nodes = np.zeros((n_elem + 1, 3))
        nodes[:, 2] = np.linspace(0, L, n_elem + 1)
        elements = []
        for i in range(n_elem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': None})
        bcs = {0: [True] * 6}
        loads = {n_elem: [0, 0, -P, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr_computed = lambda_cr * P
        error = abs(P_cr_computed - P_cr_euler) / P_cr_euler
        errors.append(error)
    rates = np.log2(errors[:-1] / errors[1:])
    assert np.all(rates > 1.9)
    assert errors[-1] < 0.0001