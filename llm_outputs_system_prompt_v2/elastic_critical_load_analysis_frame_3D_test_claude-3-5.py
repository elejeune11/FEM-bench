def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Verify critical loads match Euler theory for circular cantilever parameter sweep."""
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
            x = np.linspace(0, 0, n_elements + 1)
            y = np.linspace(0, 0, n_elements + 1)
            z = np.linspace(0, L, n_elements + 1)
            nodes = np.column_stack([x, y, z])
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
            bcs = {0: [True] * 6}
            P = 1000.0
            loads = {n_elements: [0, P, 0, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_cr_num = lambda_cr * P
            assert_allclose(P_cr_num, P_cr_euler, rtol=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Verify critical load and mode shape invariance under rigid rotation."""
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    b = 0.1
    h = 0.2
    n_elements = 10
    x = np.linspace(0, 0, n_elements + 1)
    y = np.linspace(0, 0, n_elements + 1)
    z = np.linspace(0, L, n_elements + 1)
    nodes_base = np.column_stack([x, y, z])
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = min(Iy, Iz) * 0.9
    I_rho = Iy + Iz
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    P = 1000.0
    loads = {n_elements: [0, P, 0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads)
    rotation = Rotation.from_euler('xyz', [30, 45, 60], degrees=True)
    R = rotation.as_matrix()
    nodes_rot = (R @ nodes_base.T).T
    elements_rot = []
    for i in range(n_elements):
        ele = elements_base[i].copy()
        if ele['local_z'] is not None:
            ele['local_z'] = R @ ele['local_z']
        elements_rot.append(ele)
    loads_rot = {n_elements: np.concatenate([R @ [0, P, 0], R @ [0, 0, 0]])}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    T = np.zeros((6 * (n_elements + 1), 6 * (n_elements + 1)))
    for i in range(n_elements + 1):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert_allclose(lambda_rot, lambda_base, rtol=1e-10)
    scale = np.sum(mode_rot * (T @ mode_base)) / np.sum((T @ mode_base) ** 2)
    assert_allclose(mode_rot, scale * (T @ mode_base), rtol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify convergence to Euler buckling load with mesh refinement."""
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    r = 0.05
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P = 1000.0
    P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    relative_errors = []
    for n_elements in n_elements_list:
        x = np.linspace(0, 0, n_elements + 1)
        y = np.linspace(0, 0, n_elements + 1)
        z = np.linspace(0, L, n_elements + 1)
        nodes = np.column_stack([x, y, z])
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
        bcs = {0: [True] * 6}
        loads = {n_elements: [0, P, 0, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr_num = lambda_cr * P
        relative_errors.append(abs(P_cr_num - P_cr_euler) / P_cr_euler)
    for i in range(len(relative_errors) - 1):
        ratio = relative_errors[i] / relative_errors[i + 1]
        assert ratio > 3.5
    assert relative_errors[-1] < 1e-08