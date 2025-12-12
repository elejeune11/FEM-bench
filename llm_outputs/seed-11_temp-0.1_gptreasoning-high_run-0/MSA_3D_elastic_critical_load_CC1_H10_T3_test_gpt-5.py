def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements and tolerances adequate for discretization error.
    """
    E = 70000000000.0
    nu = 0.33
    P_ref = 1.0
    r_values = [0.5, 0.75, 1.0]
    L_values = [10.0, 20.0, 40.0]
    n_el = 10
    rel_tol = 0.02
    for r in r_values:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in L_values:
            n_nodes = n_el + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = z
            elements = []
            for i in range(n_el):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
            bc = {0: (1, 1, 1, 1, 1, 1)}
            loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
            (lam, mode) = fcn(node_coords, elements, bc, loads)
            assert lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            Pcr_analytical = math.pi ** 2 * E * I / (2.0 * L) ** 2
            Pcr_numerical = lam * P_ref
            rel_err = abs(Pcr_numerical - Pcr_analytical) / Pcr_analytical
            assert rel_err < rel_tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever model and again after applying a rigid-body rotation R to the geometry,
    element axes, and applied load. The critical load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
    Build T as a block-diagonal matrix with R applied to both [ux, uy, uz] and [θx, θy, θz] at each node.
    Then mode_rot ≈ T @ mode_base, allowing for arbitrary scale and sign.
    """
    E = 210000000000.0
    nu = 0.3
    L = 12.0
    n_el = 12
    P_ref = 1.0
    b = 0.4
    h = 0.6
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = b * h ** 3 / 3.0 if h <= b else h * b ** 3 / 3.0
    n_nodes = n_el + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    node_coords[:, 2] = z
    elements = []
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
    bc = {0: (1, 1, 1, 1, 1, 1)}
    loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
    (lam_base, mode_base) = fcn(node_coords, elements, bc, loads)
    assert lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)

    def rot_x(a):
        (ca, sa) = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def rot_y(a):
        (ca, sa) = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)
    R = rot_y(-0.35) @ rot_x(0.6)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for e in elements:
        elements_rot.append({**e, 'local_z': (R @ np.array(e['local_z']).reshape(3)).reshape(3)})
    loads_rot = {n_nodes - 1: tuple((R @ np.array([0.0, 0.0, -P_ref])).tolist() + [0.0, 0.0, 0.0])[:6]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc, loads_rot)
    assert lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    assert math.isclose(lam_rot, lam_base, rel_tol=1e-06, abs_tol=0.0)
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[i * 6:i * 6 + 3, i * 6:i * 6 + 3] = R
        T[i * 6 + 3:i * 6 + 6, i * 6 + 3:i * 6 + 6] = R
    a = T @ mode_base
    b = mode_rot
    denom = float(a @ a)
    assert denom > 0.0
    s = float(a @ b) / denom
    resid = np.linalg.norm(b - s * a)
    norm_b = np.linalg.norm(b)
    if norm_b == 0.0:
        assert resid < 1e-12
    else:
        assert resid / norm_b < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and
    that the finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    r = 0.6
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    P_ref = 1.0
    Pcr_analytical = math.pi ** 2 * E * I / (2.0 * L) ** 2
    n_els_list = [2, 4, 8, 16, 32]
    rel_errors = []
    for n_el in n_els_list:
        n_nodes = n_el + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        elements = []
        for i in range(n_el):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])})
        bc = {0: (1, 1, 1, 1, 1, 1)}
        loads = {n_nodes - 1: (0.0, 0.0, -P_ref, 0.0, 0.0, 0.0)}
        (lam, mode) = fcn(node_coords, elements, bc, loads)
        assert lam > 0.0
        assert mode.shape == (6 * n_nodes,)
        Pcr_numerical = lam * P_ref
        rel_err = abs(Pcr_numerical - Pcr_analytical) / Pcr_analytical
        rel_errors.append(rel_err)
    assert rel_errors[-1] < rel_errors[0]
    for k in range(1, len(rel_errors)):
        assert rel_errors[k] <= rel_errors[k - 1] * 1.02
    assert rel_errors[-1] < 0.005