def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 1e-2.
    """

    def build_model(L, r, n_el, P_ref):
        n_nodes = n_el + 1
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        for i in range(n_nodes):
            node_coords[i, 2] = i * (L / n_el)
        E = 210000000000.0
        nu = 0.3
        A = math.pi * r ** 2
        Iy = math.pi * r ** 4 / 4.0
        Iz = Iy
        J = math.pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        elements = []
        for i in range(n_el):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        bc = {0: [True, True, True, True, True, True]}
        loads = {n_el: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        return (node_coords, elements, bc, loads, E, Iy)

    def euler_cantilever(E, I, L):
        K = 2.0
        return math.pi ** 2 * E * I / (K * L) ** 2
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_el = 10
    P_ref = 1000.0
    for r in radii:
        for L in lengths:
            (node_coords, elements, bc, loads, E, I) = build_model(L, r, n_el, P_ref)
            (lam, mode) = fcn(node_coords, elements, bc, loads)
            Pcr_num = lam * P_ref
            Pcr_ref = euler_cantilever(E, I, L)
            rel_err = abs(Pcr_num - Pcr_ref) / Pcr_ref
            assert rel_err < 0.02

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """

    def rot_z(a):
        (ca, sa) = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def rot_y(a):
        (ca, sa) = (math.cos(a), math.sin(a))
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=float)

    def build_rect_model(L, b, h, n_el, P_ref, R=None):
        n_nodes = n_el + 1
        coords = np.zeros((n_nodes, 3), dtype=float)
        for i in range(n_nodes):
            coords[i, 2] = i * (L / n_el)
        E = 70000000000.0
        nu = 0.29
        A = b * h
        Iy = b * h ** 3 / 12.0
        Iz = h * b ** 3 / 12.0
        J = Iy + Iz
        I_rho = Iy + Iz
        elements = []
        for i in range(n_el):
            local_z = np.array([1.0, 0.0, 0.0], dtype=float)
            if R is not None:
                local_z = R @ local_z
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z.tolist()})
        if R is None:
            node_coords = coords
        else:
            node_coords = coords @ R.T
        bc = {0: [True, True, True, True, True, True]}
        base_force = np.array([0.0, 0.0, -P_ref], dtype=float)
        base_moment = np.zeros(3, dtype=float)
        if R is None:
            load_vec = np.r_[base_force, base_moment]
        else:
            load_vec = np.r_[R @ base_force, R @ base_moment]
        loads = {n_el: load_vec.tolist()}
        return (node_coords, elements, bc, loads)

    def build_T(R, n_nodes):
        T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
        for i in range(n_nodes):
            i6 = 6 * i
            T[i6:i6 + 3, i6:i6 + 3] = R
            T[i6 + 3:i6 + 6, i6 + 3:i6 + 6] = R
        return T
    L = 12.0
    b = 0.2
    h = 0.1
    n_el = 12
    P_ref = 2500.0
    (node_coords_base, elements_base, bc_base, loads_base) = build_rect_model(L, b, h, n_el, P_ref, R=None)
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, bc_base, loads_base)
    R = rot_z(0.3) @ rot_y(0.7)
    (node_coords_rot, elements_rot, bc_rot, loads_rot) = build_rect_model(L, b, h, n_el, P_ref, R=R)
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc_rot, loads_rot)
    rel_diff = abs(lam_rot - lam_base) / max(1.0, abs(lam_base))
    assert rel_diff < 1e-08
    n_nodes = n_el + 1
    T = build_T(R, n_nodes)
    mode_pred = T @ mode_base
    denom = float(np.dot(mode_pred, mode_pred))
    if denom == 0.0:
        pytest.fail('Predicted mode has zero norm.')
    alpha = float(np.dot(mode_rot, mode_pred)) / denom
    resid = np.linalg.norm(mode_rot - alpha * mode_pred) / max(1e-14, np.linalg.norm(mode_rot))
    assert resid < 5e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """

    def build_model(L, r, n_el, P_ref):
        n_nodes = n_el + 1
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        for i in range(n_nodes):
            node_coords[i, 2] = i * (L / n_el)
        E = 210000000000.0
        nu = 0.3
        A = math.pi * r ** 2
        Iy = math.pi * r ** 4 / 4.0
        Iz = Iy
        J = math.pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        elements = []
        for i in range(n_el):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        bc = {0: [True, True, True, True, True, True]}
        loads = {n_el: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        return (node_coords, elements, bc, loads, E, Iy)

    def euler_cantilever(E, I, L):
        K = 2.0
        return math.pi ** 2 * E * I / (K * L) ** 2
    L = 20.0
    r = 1.0
    P_ref = 1000.0
    n_el_list = [2, 4, 8, 16, 32]
    errors = []
    Pcr_ref = None
    for n_el in n_el_list:
        (node_coords, elements, bc, loads, E, I) = build_model(L, r, n_el, P_ref)
        (lam, mode) = fcn(node_coords, elements, bc, loads)
        Pcr_num = lam * P_ref
        if Pcr_ref is None:
            Pcr_ref = euler_cantilever(E, I, L)
        rel_err = abs(Pcr_num - Pcr_ref) / Pcr_ref
        errors.append(rel_err)
    assert errors[-1] < errors[0] / 2.0
    assert errors[-1] < errors[-2]
    assert errors[-1] < 0.01