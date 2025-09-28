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
            nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': 2 * I, 'local_z': [1, 0, 0]})
            bcs = {0: [True] * 6}
            loads = {n_elements: [0, 1, 0, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_num = lambda_cr
            rel_error = abs(P_num - P_euler) / P_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Test orientation invariance of buckling solution for rectangular section cantilever."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    b = 0.05
    h = 0.1
    n_elements = 10
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = min(Iy, Iz) / 3
    I_rho = Iy + Iz
    nodes_base = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1, 0, 0]})
    bcs = {0: [True] * 6}
    loads_base = {n_elements: [0, 1, 0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    R = Rotation.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    nodes_rot = (R @ nodes_base.T).T
    elements_rot = []
    for e in elements_base:
        e_rot = e.copy()
        e_rot['local_z'] = R @ np.array(e['local_z'])
        elements_rot.append(e_rot)
    loads_rot = {n_elements: (R @ np.array(loads_base[n_elements])).tolist()}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    n_nodes = len(nodes_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert abs(lambda_rot - lambda_base) / lambda_base < 1e-10
    mode_base_rot = T @ mode_base
    scale = np.dot(mode_rot, mode_base_rot) / np.dot(mode_base_rot, mode_base_rot)
    assert np.allclose(mode_rot, scale * mode_base_rot, rtol=1e-10, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Test mesh convergence to analytical Euler buckling load for circular cantilever."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    r = 0.5
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32]
    errors = []
    for n in n_elements_list:
        nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n + 1)])
        elements = []
        for i in range(n):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1, 0, 0]})
        bcs = {0: [True] * 6}
        loads = {n: [0, 1, 0, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        rel_error = abs(lambda_cr - P_euler) / P_euler
        errors.append(rel_error)
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 3.5
    assert errors[-1] < 1e-08