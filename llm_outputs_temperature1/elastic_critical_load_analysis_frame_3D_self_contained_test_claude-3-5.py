def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Verify critical loads match Euler buckling theory for cantilever circular columns 
    across parameter sweep of radius and length.
    """
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
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I, 'local_z': None})
            bcs = {0: [True] * 6}
            P = -1000.0
            loads = {n_elements: [0, 0, P, 0, 0, 0]}
            (lambda_cr, mode) = fcn(nodes, elements, bcs, loads)
            P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            P_cr_num = -lambda_cr * P
            assert np.abs(P_cr_num / P_cr_euler - 1.0) < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Verify critical load and mode shapes are invariant under rigid rotation 
    for rectangular section cantilever.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    L = 10.0
    n_elements = 10
    nodes_base = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
    E = 200000000000.0
    nu = 0.3
    b = 0.05
    h = 0.1
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = b * h * (b ** 2 + h ** 2) / 12
    I_rho = Iy + Iz
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    P = -1000.0
    loads_base = {n_elements: [0, 0, P, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    R = Rotation.from_euler('y', 45, degrees=True).as_matrix()
    nodes_rot = (R @ nodes_base.T).T
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': R @ [0, 1, 0]})
    loads_rot = {n_elements: R @ [0, 0, P, 0, 0, 0]}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    T = np.zeros((6 * (n_elements + 1), 6 * (n_elements + 1)))
    for i in range(n_elements + 1):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert np.abs(lambda_rot / lambda_base - 1.0) < 1e-10
    scale = np.sum(mode_rot * (T @ mode_base)) / np.sum((T @ mode_base) ** 2)
    assert np.allclose(mode_rot, scale * (T @ mode_base), rtol=1e-10, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify convergence to analytical Euler buckling load with mesh refinement
    for circular cantilever.
    """
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    r = 0.5
    L = 20.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    P = -1000.0
    P_cr_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    rel_errors = []
    for n in n_elements_list:
        nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n + 1)])
        elements = []
        for i in range(n):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I, 'local_z': None})
        bcs = {0: [True] * 6}
        loads = {n: [0, 0, P, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr_num = -lambda_cr * P
        rel_errors.append(abs(P_cr_num / P_cr_euler - 1.0))
    rates = np.log2(rel_errors[:-1]) - np.log2(rel_errors[1:])
    assert np.all(rates > 1.9)
    assert rel_errors[-1] < 1e-06