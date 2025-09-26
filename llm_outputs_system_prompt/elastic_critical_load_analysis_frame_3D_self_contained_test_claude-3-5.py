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
            nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': 2 * I, 'I_rho': 2 * I, 'local_z': [0, 1, 0]})
            bcs = {0: [True] * 6}
            P = 1000.0
            loads = {n_elements: [0, 0, -P, 0, 0, 0]}
            (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
            P_cr = lambda_cr * P
            P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
            rel_error = abs(P_cr - P_euler) / P_euler
            assert rel_error < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Verify buckling analysis is invariant under rigid rotation of the structure."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    b = 0.1
    h = 0.2
    n_elements = 10
    nodes_base = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': b * h, 'Iy': b * h ** 3 / 12, 'Iz': h * b ** 3 / 12, 'J': b * h * (b ** 2 + h ** 2) / 12, 'I_rho': b * h * (b ** 2 + h ** 2) / 12, 'local_z': [0, 1, 0]})
    bcs = {0: [True] * 6}
    P = 1000.0
    loads_base = {n_elements: [0, 0, -P, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs, loads_base)
    R = Rotation.from_euler('yx', [30, 45], degrees=True).as_matrix()
    nodes_rot = (R @ nodes_base.T).T
    loads_rot = {n_elements: (R @ np.array([0, 0, -P, 0, 0, 0])).tolist()}
    elements_rot = []
    for i in range(n_elements):
        el = elements_base[i].copy()
        el['local_z'] = (R @ np.array([0, 1, 0])).tolist()
        elements_rot.append(el)
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs, loads_rot)
    n_nodes = len(nodes_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    assert abs(lambda_base - lambda_rot) / lambda_base < 1e-10
    mode_rot_transformed = T @ mode_base
    scale = np.linalg.norm(mode_rot) / np.linalg.norm(mode_rot_transformed)
    assert np.allclose(mode_rot, scale * mode_rot_transformed, rtol=1e-10, atol=1e-10) or np.allclose(mode_rot, -scale * mode_rot_transformed, rtol=1e-10, atol=1e-10)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify convergence to analytical Euler buckling load with mesh refinement."""
    import numpy as np
    E = 200000000000.0
    nu = 0.3
    L = 10.0
    r = 0.05
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    P = 1000.0
    P_euler = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [4, 8, 16, 32, 64]
    errors = []
    for n_elements in n_elements_list:
        nodes = np.array([[0, 0, z] for z in np.linspace(0, L, n_elements + 1)])
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': 2 * I, 'I_rho': 2 * I, 'local_z': [0, 1, 0]})
        bcs = {0: [True] * 6}
        loads = {n_elements: [0, 0, -P, 0, 0, 0]}
        (lambda_cr, _) = fcn(nodes, elements, bcs, loads)
        P_cr = lambda_cr * P
        errors.append(abs(P_cr - P_euler) / P_euler)
    convergence_rates = np.log2(errors[:-1]) - np.log2(errors[1:])
    assert np.all(convergence_rates > 1.9)
    assert errors[-1] < 1e-06