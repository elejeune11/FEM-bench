def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Test Euler buckling of cantilever circular columns with parameter sweep.
    Verifies critical loads against analytical solutions across multiple geometries.
    """
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elements = 10
    for r in radii:
        for L in lengths:
            nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = 2 * I
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I, 'local_z': [0, 1, 0]})
            bcs = {0: [True] * 6}
            P = 1.0
            loads = {n_elements: [P, 0, 0, 0, 0, 0]}
            (lambda_cr, mode) = fcn(nodes, elements, bcs, loads)
            P_cr = lambda_cr * P
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            rel_error = abs(P_cr - P_euler) / P_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Test orientation invariance of buckling analysis for rectangular section cantilever.
    Verifies critical load and mode shapes are consistent after rigid rotation.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    L = 10.0
    n_elements = 10
    nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
    E = 200000000000.0
    nu = 0.3
    h = 0.1
    b = 0.05
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    I_rho = Iy + Iz
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    P = 1.0
    loads = {n_elements: [P, 0, 0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes, elements, bcs, loads)
    R = Rotation.from_euler('y', 45, degrees=True).as_matrix()
    nodes_rot = nodes @ R.T
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': R @ [0, 1, 0]})
    loads_rot = {n_elements: (R @ [P, 0, 0, 0, 0, 0]).tolist()}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    assert abs(lambda_rot - lambda_base) / lambda_base < 1e-10
    T = np.zeros((6 * (n_elements + 1), 6 * (n_elements + 1)))
    for i in range(n_elements + 1):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_rot = T @ mode_base
    scale = np.linalg.norm(mode_rot) / np.linalg.norm(mode_base_rot)
    if np.dot(mode_rot, mode_base_rot) < 0:
        scale *= -1
    assert np.allclose(mode_rot, scale * mode_base_rot, rtol=1e-10, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Test mesh convergence of cantilever Euler buckling analysis.
    Verifies error decreases with mesh refinement and achieves high accuracy.
    """
    import numpy as np
    L = 10.0
    r = 0.5
    E = 200000000000.0
    nu = 0.3
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    P = 1.0
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    errors = []
    for n in n_elements_list:
        nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n + 1)])
        elements = []
        for i in range(n):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I, 'local_z': [0, 1, 0]})
        bcs = {0: [True] * 6}
        loads = {n: [P, 0, 0, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr = lambda_cr * P
        errors.append(abs(P_cr - P_euler) / P_euler)
    rates = np.log(errors[:-1]) / np.log(errors[1:])
    assert np.all(rates > 1.9)
    assert errors[-1] < 1e-06