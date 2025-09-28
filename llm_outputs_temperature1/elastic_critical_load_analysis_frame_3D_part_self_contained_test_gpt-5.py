def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements. Tolerances set for anticipated discretization error.
    """
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    ne = 10
    E = 70000000000.0
    nu = 0.3
    local_z_dir = np.array([1.0, 0.0, 0.0])
    F_ref = 1.0
    rtol = 0.002
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = np.pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        for L in lengths:
            n_nodes = ne + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = z
            elements = []
            for i in range(ne):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_dir.tolist()})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -F_ref, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
            assert np.isfinite(lam) and lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam * F_ref - P_euler) / P_euler
            assert rel_err < rtol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 70000000000.0
    nu = 0.3
    L = 12.0
    ne = 12
    b_y = 0.8
    b_z = 1.2
    A = b_y * b_z
    Iy = b_y * b_z ** 3 / 12.0
    Iz = b_z * b_y ** 3 / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    local_z_dir = np.array([1.0, 0.0, 0.0])
    n_nodes = ne + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.zeros((n_nodes, 3), dtype=float)
    node_coords_base[:, 2] = z
    elements_base = []
    for i in range(ne):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_dir.tolist()})
    boundary_conditions = {0: [True, True, True, True, True, True]}
    F_ref = 1.0
    load_base = np.array([0.0, 0.0, -F_ref, 0.0, 0.0, 0.0])
    nodal_loads_base = {n_nodes - 1: load_base.tolist()}
    (lam_base, mode_base) = fcn(node_coords=node_coords_base, elements=elements_base, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_base)

    def Rx(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])

    def Ry(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])

    def Rz(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    R = Rz(np.deg2rad(20.0)) @ Ry(np.deg2rad(35.0)) @ Rx(np.deg2rad(15.0))
    node_coords_rot = (R @ node_coords_base.T).T
    elements_rot = []
    for e in elements_base:
        e_rot = dict(e)
        e_rot['local_z'] = (R @ local_z_dir).tolist()
        elements_rot.append(e_rot)
    load_rot = R @ load_base[:3]
    nodal_loads_rot = {n_nodes - 1: [load_rot[0], load_rot[1], load_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords=node_coords_rot, elements=elements_rot, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_rot)
    assert np.isfinite(lam_base) and np.isfinite(lam_rot)
    assert lam_base > 0.0 and lam_rot > 0.0
    assert np.isclose(lam_base, lam_rot, rtol=1e-08, atol=0.0)
    N = n_nodes
    T = np.zeros((6 * N, 6 * N), dtype=float)
    for a in range(N):
        T[6 * a + 0:6 * a + 3, 6 * a + 0:6 * a + 3] = R
        T[6 * a + 3:6 * a + 6, 6 * a + 3:6 * a + 6] = R
    v = T @ mode_base
    w = mode_rot
    denom = np.dot(v, v)
    if denom == 0.0:
        assert False
    s = np.dot(v, w) / denom
    res = np.linalg.norm(w - s * v) / (np.linalg.norm(w) + 1e-30)
    assert res < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 70000000000.0
    nu = 0.3
    r = 0.6
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = np.pi * r ** 4 / 2.0
    I_rho = Iy + Iz
    L = 20.0
    local_z_dir = np.array([1.0, 0.0, 0.0])
    F_ref = 1.0
    ne_list = [4, 8, 16, 32]
    P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    errors = []
    for ne in ne_list:
        n_nodes = ne + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        elements = []
        for i in range(ne):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_dir.tolist()})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -F_ref, 0.0, 0.0, 0.0]}
        (lam, _) = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
        err = abs(lam * F_ref - P_euler) / P_euler
        errors.append(err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i] - 1e-08
    assert errors[-1] < 0.0005