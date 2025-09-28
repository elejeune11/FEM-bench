def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Test Euler buckling of cantilever circular columns across parameter ranges.
    Verifies critical loads match analytical solutions within tolerance."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elements = 10
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = 2 * I
            I_rho = 2 * I
            z = np.linspace(0, L, n_elements + 1)
            nodes = np.zeros((n_elements + 1, 3))
            nodes[:, 2] = z
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': None})
            bcs = {0: [True] * 6}
            P = 1000.0
            loads = {n_elements: [0, 0, -P, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_numerical = lambda_cr * P
            rel_error = abs(P_numerical - P_euler) / P_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Test orientation invariance of buckling analysis with rectangular section."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    h = 0.1
    b = 0.05
    L = 2.0
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = min(Iy, Iz) / 3
    I_rho = Iy + Iz
    nodes_base = np.array([[0, 0, 0], [0, 0, L / 3], [0, 0, 2 * L / 3], [0, 0, L]])
    elements_base = []
    for i in range(3):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    P = 1000.0
    loads = {3: [0, 0, -P, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads)
    theta = np.pi / 4
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    nodes_rot = nodes_base @ R.T
    local_z_rot = R @ np.array([0, 1, 0])
    elements_rot = []
    for i in range(3):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    loads_rot = {3: (R @ np.array([0, 0, -P, 0, 0, 0])).tolist()}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    n_nodes = len(nodes_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert abs(lambda_rot - lambda_base) < 1e-10
    mode_rot_scaled = mode_rot / np.max(np.abs(mode_rot))
    mode_base_transformed = T @ (mode_base / np.max(np.abs(mode_base)))
    assert np.allclose(mode_rot_scaled, mode_base_transformed, atol=1e-10) or np.allclose(mode_rot_scaled, -mode_base_transformed, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Test mesh convergence for Euler buckling of a circular cantilever."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    r = 0.05
    L = 4.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32]
    rel_errors = []
    for n in n_elements_list:
        z = np.linspace(0, L, n + 1)
        nodes = np.zeros((n + 1, 3))
        nodes[:, 2] = z
        elements = []
        for i in range(n):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': None})
        bcs = {0: [True] * 6}
        P = 1000.0
        loads = {n: [0, 0, -P, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_numerical = lambda_cr * P
        rel_errors.append(abs(P_numerical - P_euler) / P_euler)
    for i in range(len(rel_errors) - 1):
        ratio = rel_errors[i] / rel_errors[i + 1]
        assert ratio > 3.5
    assert rel_errors[-1] < 1e-08