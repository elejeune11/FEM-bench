def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements and tolerances appropriate for discretization at this mesh density.
    """
    E = 1.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    P_ref = 1.0
    for r in radii:
        A = np.pi * r ** 2
        Iy = np.pi * r ** 4 / 4.0
        Iz = Iy
        J = np.pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        for L in lengths:
            z = np.linspace(0.0, L, n_elems + 1)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0.0, 1.0, 0.0]})
            bc = {0: [True, True, True, True, True, True]}
            loads = {n_elems: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, bc, loads)
            assert mode.shape == (6 * (n_elems + 1),)
            P_cr_euler = np.pi ** 2 * E * Iy / (4.0 * L ** 2)
            rel_err = abs(lam - P_cr_euler) / P_cr_euler
            assert rel_err < 0.03

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R.
    """
    E = 2.0
    nu = 0.3
    b = 1.0
    h = 2.0
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    L = 15.0
    n_elems = 12
    z = np.linspace(0.0, L, n_elems + 1)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements = []
    local_z = np.array([1.0, 0.0, 0.0])
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z.tolist()})
    bc = {0: [True, True, True, True, True, True]}
    P_ref = 1.0
    loads = {n_elems: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, bc, loads)
    assert mode_base.shape == (6 * (n_elems + 1),)
    alpha = np.deg2rad(30.0)
    beta = np.deg2rad(-20.0)
    gamma = np.deg2rad(40.0)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(alpha), -np.sin(alpha)], [0.0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0.0, np.sin(beta)], [0.0, 1.0, 0.0], [-np.sin(beta), 0.0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0.0], [np.sin(gamma), np.cos(gamma), 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ Ry @ Rx
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        e_rot = dict(e)
        e_rot['local_z'] = (R @ local_z).tolist()
        elements_rot.append(e_rot)
    f_global = np.array([0.0, 0.0, -P_ref])
    f_rot = R @ f_global
    loads_rot = {n_elems: [f_rot[0], f_rot[1], f_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc, loads_rot)
    assert mode_rot.shape == (6 * (n_elems + 1),)
    rel_diff_lam = abs(lam_rot - lam_base) / abs(lam_base)
    assert rel_diff_lam < 1e-09
    n_nodes = n_elems + 1
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_rotated = T @ mode_base
    denom = np.dot(mode_base_rotated, mode_base_rotated)
    if denom == 0.0:
        scale = 1.0
    else:
        scale = np.dot(mode_base_rotated, mode_rot) / denom
    res = mode_rot - scale * mode_base_rotated
    rel_mode_err = np.linalg.norm(res) / max(np.linalg.norm(mode_rot), 1e-16)
    assert rel_mode_err < 1e-08

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 1.0
    nu = 0.3
    r = 0.9
    A = np.pi * r ** 2
    Iy = np.pi * r ** 4 / 4.0
    Iz = Iy
    J = np.pi * r ** 4 / 2.0
    I_rho = Iy + Iz
    L = 25.0
    P_ref = 1.0
    meshes = [2, 4, 8, 16, 32]
    errors = []
    for n_elems in meshes:
        z = np.linspace(0.0, L, n_elems + 1)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0.0, 1.0, 0.0]})
        bc = {0: [True, True, True, True, True, True]}
        loads = {n_elems: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, bc, loads)
        assert mode.shape == (6 * (n_elems + 1),)
        P_cr_euler = np.pi ** 2 * E * Iy / (4.0 * L ** 2)
        rel_err = abs(lam - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1] + 1e-12
    assert errors[-1] < 0.001