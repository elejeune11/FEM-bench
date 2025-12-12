def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements; check that the relative error is small for this discretization.
    """
    E = 210000000000.0
    nu = 0.3
    ne = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    tol = 0.005
    for r in radii:
        A = np.pi * r ** 2
        Iy = Iz = np.pi * r ** 4 / 4.0
        J = Iy + Iz
        for L in lengths:
            n_nodes = ne + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = z
            elements = []
            for i in range(ne):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            Pcr_num = lam * 1.0
            Pcr_euler = np.pi ** 2 * E * Iy / (4.0 * L ** 2)
            rel_err = abs(Pcr_num - Pcr_euler) / Pcr_euler
            assert rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 200000000000.0
    nu = 0.3
    L = 12.0
    ne = 12
    n_nodes = ne + 1
    b = 2.0
    h = 1.0
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = z
    elements_base = []
    for i in range(ne):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)

    def rot_x(a):
        (c, s) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(a):
        (c, s) = (np.cos(a), np.sin(a))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Rx = rot_x(np.deg2rad(30.0))
    Ry = rot_y(np.deg2rad(20.0))
    R = Ry @ Rx
    node_coords_rot = node_coords_base @ R.T
    elements_rot = []
    base_local_z = np.array([0.0, 1.0, 0.0])
    local_z_rot = R @ base_local_z
    for i in range(ne):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_rot})
    tip_force_base = np.array([0.0, 0.0, -1.0])
    tip_force_rot = R @ tip_force_base
    nodal_loads_rot = {n_nodes - 1: [tip_force_rot[0], tip_force_rot[1], tip_force_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff_lam = abs(lam_rot - lam_base) / abs(lam_base)
    assert rel_diff_lam < 1e-08
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_pred = T @ mode_base
    nb = np.linalg.norm(mode_pred)
    nr = np.linalg.norm(mode_rot)
    if nb == 0 or nr == 0:
        raise AssertionError('Returned mode vector has zero norm.')
    mode_pred_n = mode_pred / nb
    mode_rot_n = mode_rot / nr
    corr = abs(np.dot(mode_pred_n, mode_rot_n))
    assert corr > 1.0 - 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    L = 20.0
    r = 1.0
    A = np.pi * r ** 2
    Iy = Iz = np.pi * r ** 4 / 4.0
    J = Iy + Iz
    meshes = [2, 4, 8, 16]
    errors = []
    for ne in meshes:
        n_nodes = ne + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = z
        elements = []
        for i in range(ne):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        Pcr_num = lam * 1.0
        Pcr_euler = np.pi ** 2 * E * Iy / (4.0 * L ** 2)
        rel_err = abs(Pcr_num - Pcr_euler) / Pcr_euler
        errors.append(rel_err)
    for k in range(1, len(errors)):
        assert errors[k] < errors[k - 1]
    assert errors[-1] < 0.001