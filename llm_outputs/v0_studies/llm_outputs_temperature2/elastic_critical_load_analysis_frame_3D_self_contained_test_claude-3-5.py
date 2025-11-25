def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Test Euler buckling critical loads for cantilever circular columns across parameter sweep."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elements = 10
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4
        J = 2 * I
        for L in lengths:
            x = np.linspace(0, L, n_elements + 1)
            nodes = np.zeros((n_elements + 1, 3))
            nodes[:, 2] = x
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': 2 * I, 'local_z': [0, 1, 0]})
            bcs = {0: [True] * 6}
            loads = {n_elements: [0, 0, -1000, 0, 0, 0]}
            (lambda_cr, mode) = fcn(nodes, elements, bcs, loads)
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_num = lambda_cr * 1000
            assert abs(P_num / P_euler - 1.0) < 0.0001

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Test invariance of buckling solution under rigid rotation of the model."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    L = 10.0
    n_elements = 10
    E = 200000000000.0
    nu = 0.3
    b = 0.05
    h = 0.1
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    I_rho = Iy + Iz
    nodes_base = np.zeros((n_elements + 1, 3))
    nodes_base[:, 2] = np.linspace(0, L, n_elements + 1)
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    loads = {n_elements: [0, 0, -1000, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads)
    theta = np.pi / 3
    R = Rotation.from_rotvec([1, 1, 1] * theta / np.sqrt(3)).as_matrix()
    nodes_rot = (R @ nodes_base.T).T
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': R @ [0, 1, 0]})
    loads_rot = {n_elements: np.concatenate((R @ [0, 0, -1000], R @ [0, 0, 0]))}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    assert abs(lambda_rot / lambda_base - 1.0) < 1e-10
    T = np.zeros((6 * (n_elements + 1), 6 * (n_elements + 1)))
    for i in range(n_elements + 1):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_norm = mode_base / np.max(np.abs(mode_base))
    mode_rot_norm = mode_rot / np.max(np.abs(mode_rot))
    assert np.allclose(mode_rot_norm, T @ mode_base_norm, atol=1e-10) or np.allclose(mode_rot_norm, -T @ mode_base_norm, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Test mesh convergence to analytical Euler buckling load."""
    import numpy as np
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    r = 0.05
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32]
    errors = []
    for n in n_elements_list:
        nodes = np.zeros((n + 1, 3))
        nodes[:, 2] = np.linspace(0, L, n + 1)
        elements = []
        for i in range(n):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': 2 * I, 'local_z': [0, 1, 0]})
        bcs = {0: [True] * 6}
        loads = {n: [0, 0, -1000, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_num = lambda_cr * 1000
        errors.append(abs(P_num / P_euler - 1.0))
    rates = np.log2(errors[:-1] / errors[1:])
    assert np.all(rates > 1.9)
    assert errors[-1] < 1e-05