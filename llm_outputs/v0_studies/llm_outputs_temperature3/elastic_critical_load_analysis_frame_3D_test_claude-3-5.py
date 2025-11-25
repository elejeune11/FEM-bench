def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Tests Euler buckling of cantilever circular columns against analytical solution.
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
            loads = {n_elements: [0, 0, -P, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_numerical = lambda_cr * P
            rel_error = abs(P_numerical - P_euler) / P_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Tests invariance of buckling results under rigid rotation of the structure.
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
    J = min(Iy, Iz) / 2
    I_rho = Iy + Iz
    x = np.linspace(0, L, n_elements + 1)
    nodes_base = np.zeros((n_elements + 1, 3))
    nodes_base[:, 0] = x
    elements_base = []
    for i in range(n_elements):
        ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 0, 1]}
        elements_base.append(ele)
    bcs = {0: [True] * 6}
    P = 1000.0
    loads_base = {n_elements: [0, 0, -P, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    angles = [30, 45, 60]
    for theta in angles:
        R = Rotation.from_euler('y', theta, degrees=True).as_matrix()
        nodes_rot = nodes_base @ R.T
        elements_rot = []
        for ele in elements_base:
            ele_rot = ele.copy()
            ele_rot['local_z'] = R @ np.array([0, 0, 1])
            elements_rot.append(ele_rot)
        loads_rot = {n_elements: R @ np.array([0, 0, -P, 0, 0, 0])}
        (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
        assert abs(lambda_rot - lambda_base) / lambda_base < 1e-10
        n_nodes = len(nodes_base)
        T = np.zeros((6 * n_nodes, 6 * n_nodes))
        for i in range(n_nodes):
            T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
            T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
        mode_rot_normalized = mode_rot / np.max(np.abs(mode_rot))
        mode_base_transformed = T @ mode_base
        mode_base_transformed_normalized = mode_base_transformed / np.max(np.abs(mode_base_transformed))
        assert np.allclose(mode_rot_normalized, mode_base_transformed_normalized, atol=1e-10) or np.allclose(mode_rot_normalized, -mode_base_transformed_normalized, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Tests convergence of numerical solution to analytical Euler buckling load
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
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    errors = []
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
        loads = {n_elements: [0, 0, -P, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_numerical = lambda_cr * P
        rel_error = abs(P_numerical - P_euler) / P_euler
        errors.append(rel_error)
    for i in range(len(errors) - 1):
        ratio = errors[i] / errors[i + 1]
        assert ratio > 3.5
    assert errors[-1] < 1e-08