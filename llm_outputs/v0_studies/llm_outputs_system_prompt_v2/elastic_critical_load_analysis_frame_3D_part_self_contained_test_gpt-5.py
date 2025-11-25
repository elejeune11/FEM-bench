def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 1e-2.
    """

    def build_cantilever_mesh(L, n_elems, r, E, nu):
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = z
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = math.pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        elements = []
        for e in range(n_elems):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        return (node_coords, elements)
    E = 10000000.0
    nu = 0.3
    n_elems = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    tol_rel = 0.01
    for r in radii:
        for L in lengths:
            (node_coords, elements) = build_cantilever_mesh(L, n_elems, r, E, nu)
            bc = {0: [True, True, True, True, True, True]}
            P_mag = 1.0
            loads = {node_coords.shape[0] - 1: [0.0, 0.0, -P_mag, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, bc, loads)
            I = math.pi * r ** 4 / 4.0
            P_cr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
            assert lam > 0.0
            rel_err = abs(lam - P_cr_euler) / P_cr_euler
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """

    def build_cantilever_rect(L, n_elems, E, nu, b, h):
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = z
        A = b * h
        Iy = h * b ** 3 / 12.0
        Iz = b * h ** 3 / 12.0
        J = Iy + Iz
        I_rho = Iy + Iz
        elements = []
        for e in range(n_elems):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        return (node_coords, elements)

    def rotation_matrix(angles_deg):
        (ax, ay, az) = [math.radians(a) for a in angles_deg]
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(ax), -math.sin(ax)], [0.0, math.sin(ax), math.cos(ax)]])
        Ry = np.array([[math.cos(ay), 0.0, math.sin(ay)], [0.0, 1.0, 0.0], [-math.sin(ay), 0.0, math.cos(ay)]])
        Rz = np.array([[math.cos(az), -math.sin(az), 0.0], [math.sin(az), math.cos(az), 0.0], [0.0, 0.0, 1.0]])
        return Rz @ Ry @ Rx
    E = 10000000.0
    nu = 0.3
    L = 30.0
    n_elems = 12
    b = 2.0
    h = 1.0
    (node_coords, elements) = build_cantilever_rect(L, n_elems, E, nu, b, h)
    bc = {0: [True, True, True, True, True, True]}
    loads = {node_coords.shape[0] - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, bc, loads)
    R = rotation_matrix([25.0, -17.0, 33.0])
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for e in elements:
        local_z_rot = (R @ np.array(e['local_z']).reshape(3, 1)).ravel().tolist()
        e_rot = dict(e)
        e_rot['local_z'] = local_z_rot
        elements_rot.append(e_rot)
    loads_rot = {}
    for (k, vec) in loads.items():
        f = np.array(vec[:3])
        m = np.array(vec[3:])
        f_rot = (R @ f).tolist()
        m_rot = (R @ m).tolist()
        loads_rot[k] = f_rot + m_rot
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc, loads_rot)
    assert lam_base > 0.0 and lam_rot > 0.0
    rel_diff = abs(lam_rot - lam_base) / lam_base
    assert rel_diff < 1e-10
    n_nodes = node_coords.shape[0]
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    v_base_T = T @ mode_base
    denom = float(v_base_T @ v_base_T)
    if denom == 0.0:
        alpha = 0.0
    else:
        alpha = float(v_base_T @ mode_rot) / denom
    resid = np.linalg.norm(mode_rot - alpha * v_base_T)
    norm_rot = np.linalg.norm(mode_rot)
    if norm_rot == 0.0:
        rel_resid = resid
    else:
        rel_resid = resid / norm_rot
    assert rel_resid < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """

    def build_cantilever_mesh(L, n_elems, r, E, nu):
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = z
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        Iy = I
        Iz = I
        J = math.pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        elements = []
        for e in range(n_elems):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        return (node_coords, elements)
    E = 10000000.0
    nu = 0.3
    L = 20.0
    r = 1.0
    P_mag = 1.0
    n_list = [2, 4, 8, 16, 32]
    errors = []
    P_cr_euler = math.pi ** 2 * E * (math.pi * r ** 4 / 4.0) / (4.0 * L ** 2)
    for n_elems in n_list:
        (node_coords, elements) = build_cantilever_mesh(L, n_elems, r, E, nu)
        bc = {0: [True, True, True, True, True, True]}
        loads = {node_coords.shape[0] - 1: [0.0, 0.0, -P_mag, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, bc, loads)
        rel_err = abs(lam - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]
    assert errors[-1] < 0.005