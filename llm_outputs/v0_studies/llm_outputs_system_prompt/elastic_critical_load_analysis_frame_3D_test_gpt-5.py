def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z. Parametric sweep:
    radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}. For each case,
    run the full analysis and compare λ·P_ref to the Euler cantilever analytical
    solution P_cr = (π^2/4) E I / L^2. Use 10 elements and a relative tolerance
    consistent with expected discretization error.
    """

    def build_column_cantilever_circular(L, r, n_el, E=210000000000.0, nu=0.3):
        n_nodes = n_el + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        elements = []
        local_z = [1.0, 0.0, 0.0]
        for e in range(n_el):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': J, 'local_z': local_z})
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        nodal_loads = {n_el: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        return (node_coords, elements, boundary_conditions, nodal_loads, E, I)
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_el = 10
    rel_tol = 0.01
    for r in radii:
        for L in lengths:
            (node_coords, elements, bcs, loads, E, I) = build_column_cantilever_circular(L, r, n_el)
            (lam, mode) = fcn(node_coords, elements, bcs, loads)
            P_ref = 1.0
            P_cr_num = lam * P_ref
            P_cr_euler = np.pi ** 2 / 4.0 * E * I / L ** 2
            rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_err < rel_tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz). The cantilever
    is solved in base orientation and after applying a rigid-body rotation R to the
    geometry, element axes (local_z), and applied load. The critical load factor λ
    should be invariant, and the buckling mode from the rotated model should equal
    T @ mode_base up to a scalar factor, where T applies R to both translational
    and rotational DOFs at each node.
    """

    def rotation_matrix_xyz(ax, ay, az):
        (cx, sx) = (np.cos(ax), np.sin(ax))
        (cy, sy) = (np.cos(ay), np.sin(ay))
        (cz, sz) = (np.cos(az), np.sin(az))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
        return Rz @ Ry @ Rx

    def build_cantilever_rect(L, n_el, b, h, E=200000000000.0, nu=0.3):
        n_nodes = n_el + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        A = b * h
        Iy = b * h ** 3 / 12.0
        Iz = h * b ** 3 / 12.0
        J = 1.0 / 3.0 * b * h ** 3
        I_rho = Iy + Iz
        elements = []
        local_z = [1.0, 0.0, 0.0]
        for e in range(n_el):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z})
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        nodal_loads = {n_el: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        return (node_coords, elements, boundary_conditions, nodal_loads)
    L = 15.0
    n_el = 12
    b = 3.0
    h = 1.0
    (node_coords, elements, bcs, loads) = build_cantilever_rect(L, n_el, b, h)
    (lam_base, mode_base) = fcn(node_coords, elements, bcs, loads)
    (ax, ay, az) = (0.25, -0.4, 0.15)
    R = rotation_matrix_xyz(ax, ay, az)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for ele in elements:
        lz = np.array(ele['local_z'], dtype=float)
        lz_rot = (R @ lz.reshape(3, 1)).flatten().tolist()
        ele_rot = dict(ele)
        ele_rot['local_z'] = lz_rot
        elements_rot.append(ele_rot)
    loads_vec = np.array(loads[n_el], dtype=float)
    F_rot = (R @ loads_vec[:3].reshape(3, 1)).flatten()
    M_rot = (R @ loads_vec[3:6].reshape(3, 1)).flatten()
    loads_rot = {n_el: [F_rot[0], F_rot[1], F_rot[2], M_rot[0], M_rot[1], M_rot[2]]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bcs, loads_rot)
    rel_diff_lam = abs(lam_rot - lam_base) / max(1.0, abs(lam_base))
    assert rel_diff_lam < 1e-08
    n_nodes = node_coords.shape[0]
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_T = T @ mode_base
    alpha = float(np.dot(mode_base_T, mode_rot) / np.dot(mode_base_T, mode_base_T))
    resid = mode_rot - alpha * mode_base_T
    rel_resid = np.linalg.norm(resid) / max(1e-16, np.linalg.norm(mode_rot))
    assert rel_resid < 1e-08

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the discretization and check that the numerical critical load approaches
    the analytical Euler value with decreasing relative error. The finest mesh must
    achieve very high accuracy.
    """

    def build_column(L, r, n_el, E=210000000000.0, nu=0.3):
        n_nodes = n_el + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        elements = []
        local_z = [1.0, 0.0, 0.0]
        for e in range(n_el):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': J, 'local_z': local_z})
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        nodal_loads = {n_el: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        return (node_coords, elements, boundary_conditions, nodal_loads, E, I)
    L = 20.0
    r = 0.75
    mesh_sizes = [2, 4, 8, 16]
    errors = []
    for n_el in mesh_sizes:
        (node_coords, elements, bcs, loads, E, I) = build_column(L, r, n_el)
        (lam, mode) = fcn(node_coords, elements, bcs, loads)
        P_cr_euler = np.pi ** 2 / 4.0 * E * I / L ** 2
        rel_err = abs(lam - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * 1.0001
    assert errors[-1] < 0.005