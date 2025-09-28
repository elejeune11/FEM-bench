def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Test Euler buckling of cantilever circular columns across parameter sweep.
    Verifies critical loads match analytical solutions within tolerance."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elem = 10
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = 2 * I
            x = np.linspace(0, L, n_elem + 1)
            nodes = np.zeros((n_elem + 1, 3))
            nodes[:, 2] = x
            elements = []
            for i in range(n_elem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': 2 * I, 'local_z': None})
            bcs = {0: [True] * 6}
            P = 1000.0
            loads = {n_elem: [0, 0, -P, 0, 0, 0]}
            (lambda_cr, mode) = fcn(nodes, elements, bcs, loads)
            P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_cr_num = lambda_cr * P
            rel_error = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Test orientation invariance of buckling analysis with rectangular section."""
    import numpy as np
    L = 10.0
    n_elem = 10
    nodes = np.zeros((n_elem + 1, 3))
    nodes[:, 2] = np.linspace(0, L, n_elem + 1)
    E = 200000000000.0
    b = 0.05
    h = 0.1
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    I_rho = Iy + Iz
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': 0.3, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    P = 1000.0
    loads = {n_elem: [0, 0, -P, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes, elements, bcs, loads)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    nodes_rot = nodes @ R.T
    elements_rot = []
    for i in range(n_elem):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': 0.3, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': R @ [0, 1, 0]})
    loads_rot = {n_elem: R @ [0, 0, -P, 0, 0, 0]}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    n_nodes = len(nodes)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert abs(lambda_rot - lambda_base) < 1e-10
    mode_base_rot = T @ mode_base
    scale = np.linalg.norm(mode_rot) / np.linalg.norm(mode_base_rot)
    assert np.allclose(mode_rot, scale * mode_base_rot, rtol=1e-10) or np.allclose(mode_rot, -scale * mode_base_rot, rtol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Test mesh convergence for Euler buckling of circular cantilever."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    r = 0.5
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    P = 1000.0
    P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elem_list = [4, 8, 16, 32, 64]
    rel_errors = []
    for n_elem in n_elem_list:
        nodes = np.zeros((n_elem + 1, 3))
        nodes[:, 2] = np.linspace(0, L, n_elem + 1)
        elements = []
        for i in range(n_elem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': 2 * I, 'local_z': None})
        bcs = {0: [True] * 6}
        loads = {n_elem: [0, 0, -P, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr_num = lambda_cr * P
        rel_error = abs(P_cr_num - P_cr_euler) / P_cr_euler
        rel_errors.append(rel_error)
    for i in range(len(rel_errors) - 1):
        ratio = rel_errors[i] / rel_errors[i + 1]
        assert ratio > 3.5
    assert rel_errors[-1] < 1e-08