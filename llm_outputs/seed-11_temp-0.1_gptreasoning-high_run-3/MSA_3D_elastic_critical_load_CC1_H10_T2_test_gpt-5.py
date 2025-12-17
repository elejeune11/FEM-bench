def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements and a tolerance commensurate with discretization error.
    """
    E = 210000000000.0
    nu = 0.3
    n_elem = 10
    local_z = np.array([1.0, 0.0, 0.0])
    P_ref = 1.0
    for r in [0.5, 0.75, 1.0]:
        A = pi * r ** 2
        I = pi * r ** 4 / 4.0
        J = 2.0 * I
        for L in [10.0, 20.0, 40.0]:
            n_nodes = n_elem + 1
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
            elements = []
            for e in range(n_elem):
                elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            k = 2.0
            P_cr_analytical = pi ** 2 * E * I / (k * L) ** 2
            rel_err = abs(lam - P_cr_analytical) / P_cr_analytical
            assert rel_err < 0.001
            assert np.linalg.norm(mode) > 0.0

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever model in its base orientation and again after applying
    a rigid-body rotation R to geometry, element axes, and applied load.
    The critical load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R
    (up to a scalar factor), where the transform applies R to both the translational
    and rotational DOFs at each node.
    """
    E = 210000000000.0
    nu = 0.3
    L = 12.0
    n_elem = 10
    n_nodes = n_elem + 1
    b = 0.2
    h = 0.1
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = I_y + I_z
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
    local_z_base = np.array([0.0, 1.0, 0.0])
    elements_base = []
    for e in range(n_elem):
        elements_base.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_base})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1.0
    load_base = np.array([0.0, 0.0, -P_ref])
    nodal_loads_base = {n_nodes - 1: [load_base[0], load_base[1], load_base[2], 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)

    def Rx(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])

    def Ry(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])

    def Rz(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    R = Rz(0.41) @ Ry(-0.52) @ Rx(0.37)
    node_coords_rot = node_coords @ R.T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for e in range(n_elem):
        elements_rot.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    load_rot = R @ load_base
    nodal_loads_rot = {n_nodes - 1: [load_rot[0], load_rot[1], load_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff_lambda = abs(lam_rot - lam_base) / abs(lam_base)
    assert rel_diff_lambda < 1e-08
    mode_base_mat = mode_base.reshape(n_nodes, 6)
    mode_pred_mat = np.zeros_like(mode_base_mat)
    for i in range(n_nodes):
        u = mode_base_mat[i, 0:3]
        th = mode_base_mat[i, 3:6]
        mode_pred_mat[i, 0:3] = R @ u
        mode_pred_mat[i, 3:6] = R @ th
    mode_pred = mode_pred_mat.reshape(-1)
    free_mask = np.ones(6 * n_nodes, dtype=bool)
    for node_idx, bc in boundary_conditions.items():
        for j, fix in enumerate(bc):
            if fix:
                free_mask[6 * node_idx + j] = False
    v1 = mode_pred[free_mask]
    v2 = mode_rot[free_mask]
    alpha = np.dot(v1, v2) / (np.dot(v1, v1) + 1e-30)
    resid = np.linalg.norm(v2 - alpha * v1) / (np.linalg.norm(v2) + 1e-30)
    assert resid < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load
    approaches the analytical Euler value with decreasing relative error,
    and that the finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    r = 0.5
    A = pi * r ** 2
    I = pi * r ** 4 / 4.0
    J = 2.0 * I
    local_z = np.array([1.0, 0.0, 0.0])
    P_ref = 1.0
    k = 2.0
    P_cr_analytical = pi ** 2 * E * I / (k * L) ** 2
    errors = []
    elem_counts = [2, 4, 8, 16, 32]
    for n_elem in elem_counts:
        n_nodes = n_elem + 1
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = np.linspace(0.0, L, n_nodes)
        elements = []
        for e in range(n_elem):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - P_cr_analytical) / P_cr_analytical
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] * 1.02
    assert errors[-1] < 0.0005