def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Verify critical loads match Euler theory across parameter sweep of circular cantilevers."""
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
            nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
            n_nodes = len(nodes)
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': 2 * I, 'local_z': [1, 0, 0]})
            bcs = {0: [True] * 6}
            P_ref = 1000.0
            loads = {n_nodes - 1: [P_ref, 0, 0, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_numerical = lambda_cr * P_ref
            rel_error = abs(P_numerical - P_euler) / P_euler
            assert rel_error < 0.01

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Verify critical load and mode invariance under rigid rotation."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    b = 0.5
    h = 1.0
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
    P_ref = 1000.0
    loads_base = {n_elements: [P_ref, 0, 0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    nodes_rot = (R @ nodes_base.T).T
    elements_rot = []
    for e in elements_base:
        e_rot = e.copy()
        e_rot['local_z'] = R @ np.array([1, 0, 0])
        elements_rot.append(e_rot)
    loads_rot = {n_elements: list(R @ np.array([P_ref, 0, 0, 0, 0, 0]))}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    n_nodes = len(nodes_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert abs(lambda_base - lambda_rot) < 1e-10
    scale = np.linalg.norm(mode_rot) / np.linalg.norm(mode_base)
    if np.dot(mode_rot, T @ mode_base) < 0:
        scale = -scale
    assert np.allclose(mode_rot, scale * T @ mode_base, rtol=1e-10, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify convergence to analytical Euler buckling load with mesh refinement."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    r = 0.5
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    relative_errors = []
    for n_elements in n_elements_list:
        nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
        n_nodes = len(nodes)
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': 2 * I, 'local_z': [1, 0, 0]})
        bcs = {0: [True] * 6}
        P_ref = 1000.0
        loads = {n_nodes - 1: [P_ref, 0, 0, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_numerical = lambda_cr * P_ref
        relative_errors.append(abs(P_numerical - P_euler) / P_euler)
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i] > relative_errors[i + 1]
    assert relative_errors[-1] < 0.0001