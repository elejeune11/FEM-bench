def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Test Euler buckling of cantilever circular columns against analytical solutions.
    Sweeps through multiple radii and lengths to verify correct scaling behavior.
    """
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
        I_rho = 2 * I
        for L in lengths:
            z = np.linspace(0, L, n_elements + 1)
            nodes = np.zeros((n_elements + 1, 3))
            nodes[:, 2] = z
            elements = []
            for i in range(n_elements):
                ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]}
                elements.append(ele)
            bcs = {0: [True] * 6}
            P = 1000.0
            loads = {n_elements: [0, P, 0, 0, 0, 0]}
            (lambda_cr, mode) = fcn(nodes, elements, bcs, loads)
            P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_cr_num = lambda_cr * P
            rel_error = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Test invariance of buckling results under rigid rotation of the structure.
    Uses rectangular section to verify proper handling of principal axes.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    E = 200000000000.0
    nu = 0.3
    h = 0.1
    b = 0.05
    L = 5.0
    n_elements = 10
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = min(Iy, Iz) * 0.9
    I_rho = Iy + Iz
    z = np.linspace(0, L, n_elements + 1)
    nodes_base = np.zeros((n_elements + 1, 3))
    nodes_base[:, 2] = z
    elements_base = []
    for i in range(n_elements):
        ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 0, 1]}
        elements_base.append(ele)
    bcs = {0: [True] * 6}
    P = 1000.0
    loads_base = {n_elements: [0, P, 0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    theta = np.pi / 3
    R = Rotation.from_rotvec([theta, theta / 2, -theta / 3]).as_matrix()
    nodes_rot = (R @ nodes_base.T).T
    elements_rot = []
    for i in range(n_elements):
        ele = elements_base[i].copy()
        ele['local_z'] = R @ np.array([0, 0, 1])
        elements_rot.append(ele)
    loads_rot = {n_elements: R @ np.array([0, P, 0, 0, 0, 0])}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    n_nodes = len(nodes_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert abs(lambda_base - lambda_rot) / lambda_base < 1e-10
    scale = np.linalg.norm(mode_rot) / np.linalg.norm(mode_base)
    if np.dot(mode_rot, T @ mode_base) < 0:
        scale *= -1
    assert np.allclose(mode_rot, scale * T @ mode_base, rtol=1e-10, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Test convergence of numerical solution to analytical Euler buckling load
    with mesh refinement for a circular cantilever.
    """
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    r = 0.05
    L = 10.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    rel_errors = []
    for n_elements in n_elements_list:
        z = np.linspace(0, L, n_elements + 1)
        nodes = np.zeros((n_elements + 1, 3))
        nodes[:, 2] = z
        elements = []
        for i in range(n_elements):
            ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]}
            elements.append(ele)
        bcs = {0: [True] * 6}
        P = 1000.0
        loads = {n_elements: [0, P, 0, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr_num = lambda_cr * P
        rel_errors.append(abs(P_cr_num - P_cr_euler) / P_cr_euler)
    for i in range(len(rel_errors) - 1):
        ratio = rel_errors[i] / rel_errors[i + 1]
        assert ratio > 3.5
    assert rel_errors[-1] < 1e-08