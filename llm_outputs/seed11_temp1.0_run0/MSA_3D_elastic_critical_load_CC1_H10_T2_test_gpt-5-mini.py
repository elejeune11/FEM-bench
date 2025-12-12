def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever
    analytical solution. Use 10 elements.
    """
    import numpy as np
    pi = np.pi
    E = 210000000000.0
    nu = 0.3
    n_elems = 10
    for r in (0.5, 0.75, 1.0):
        A = pi * r * r
        I = pi * r ** 4 / 4.0
        J = pi * r ** 4 / 2.0
        for L in (10.0, 20.0, 40.0):
            n_nodes = n_elems + 1
            zs = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), zs])
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
            bcs = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, bcs, nodal_loads)
            assert lam > 0.0
            Pcr = pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam - Pcr) / (abs(Pcr) + 1e-18)
            assert rel_err < 0.001

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve base cantilever and rotated version. The critical load factor λ should be
    identical in both cases. The rotated mode should equal T @ mode_base up to scale/sign.
    """
    import numpy as np
    pi = np.pi
    E = 210000000000.0
    nu = 0.3
    L = 10.0
    n_elems = 12
    n_nodes = n_elems + 1
    zs = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), zs])
    b = 1.0
    h = 2.0
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = 0.5 * (b * h ** 3) / 12.0 + 0.5 * (h * b ** 3) / 12.0
    elements_base = []
    for i in range(n_elems):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, bcs, nodal_loads_base)
    assert lam_base > 0.0
    angle_x = 30.0 * pi / 180.0
    angle_y = 20.0 * pi / 180.0
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    R = Ry @ Rx
    node_coords_rot = (R @ node_coords_base.T).T
    elements_rot = []
    for el in elements_base:
        local_z_rot = (R @ np.asarray(el['local_z']).reshape(3)).tolist()
        new_el = dict(el)
        new_el['local_z'] = local_z_rot
        elements_rot.append(new_el)
    nodal_loads_rot = {}
    for (ni, load) in nodal_loads_base.items():
        f = np.asarray(load[0:3])
        m = np.asarray(load[3:6])
        f_rot = (R @ f).tolist()
        m_rot = (R @ m).tolist()
        nodal_loads_rot[ni] = [f_rot[0], f_rot[1], f_rot[2], m_rot[0], m_rot[1], m_rot[2]]
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bcs, nodal_loads_rot)
    assert lam_rot > 0.0
    rel_err = abs(lam_rot - lam_base) / (abs(lam_base) + 1e-18)
    assert rel_err < 1e-09
    dof_per_node = 6
    T_blocks = []
    for _ in range(n_nodes):
        T_block = np.zeros((6, 6))
        T_block[0:3, 0:3] = R
        T_block[3:6, 3:6] = R
        T_blocks.append(T_block)
    T = np.block([[T_blocks[i] if i == j else np.zeros((6, 6)) for j in range(n_nodes)] for i in range(n_nodes)])
    transformed = T @ mode_base
    denom = float(np.dot(transformed, transformed))
    assert denom > 0.0
    s = float(np.dot(mode_rot, transformed) / denom)
    diff_norm = np.linalg.norm(mode_rot - s * transformed)
    rel_norm = diff_norm / (np.linalg.norm(mode_rot) + 1e-18)
    assert rel_norm < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    import numpy as np
    pi = np.pi
    E = 210000000000.0
    nu = 0.3
    r = 1.0
    A = pi * r * r
    I = pi * r ** 4 / 4.0
    J = pi * r ** 4 / 2.0
    L = 20.0
    Pcr = pi ** 2 * E * I / (4.0 * L ** 2)
    mesh_elems = [2, 4, 8, 16, 32]
    errors = []
    for n_elems in mesh_elems:
        n_nodes = n_elems + 1
        zs = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), zs])
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
        bcs = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, bcs, nodal_loads)
        rel_err = abs(lam - Pcr) / (abs(Pcr) + 1e-18)
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] + 1e-08
    assert errors[-1] < 0.0001