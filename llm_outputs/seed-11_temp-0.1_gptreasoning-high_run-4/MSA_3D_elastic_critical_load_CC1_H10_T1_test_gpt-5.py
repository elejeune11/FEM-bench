def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical value.
    Use 10 elements, with tight tolerance suitable for high-accuracy formulations.
    """
    E = 210000000000.0
    nu = 0.3
    n_elem = 10
    P_ref = 1.0
    r_values = [0.5, 0.75, 1.0]
    L_values = [10.0, 20.0, 40.0]
    tol = 2e-05
    for r in r_values:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in L_values:
            n_nodes = n_elem + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(n_elem):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            P_cr_num = lam * P_ref
            P_cr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load.
    The critical load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R,
    using T = block_diag([R, R], ..., [R, R]) across all nodes (translations and rotations).
    """
    E = 210000000000.0
    nu = 0.3
    n_elem = 12
    L = 12.0
    P_ref = 1.0
    b = 0.3
    h = 0.15
    A = b * h
    I_y = h * b ** 3 / 12.0
    I_z = b * h ** 3 / 12.0
    J = 1.0 / 3.0 * b * h ** 3
    n_nodes = n_elem + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert lam_base > 0.0

    def Rx(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])

    def Ry(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])

    def Rz(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    angles = np.deg2rad([17.0, 31.0, 7.0])
    R = Rz(angles[0]) @ Ry(angles[1]) @ Rx(angles[2])
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    local_z_base = np.array([0.0, 1.0, 0.0])
    local_z_rot = (R @ local_z_base.reshape(3, 1)).flatten()
    for i in range(n_elem):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot.tolist()})
    F_ref = np.array([0.0, 0.0, -P_ref])
    F_rot = (R @ F_ref.reshape(3, 1)).flatten()
    nodal_loads_rot = {n_nodes - 1: [float(F_rot[0]), float(F_rot[1]), float(F_rot[2]), 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_rot > 0.0
    rel_diff_lambda = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lambda < 1e-08
    T_blocks = []
    for _ in range(n_nodes):
        T_blocks.append(np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = T_blocks[i]
    mode_pred = T @ mode_base
    denom = float(mode_pred @ mode_pred)
    if denom == 0.0:
        alpha = 0.0
    else:
        alpha = float(mode_rot @ mode_pred) / denom
    resid = mode_rot - alpha * mode_pred
    rel_mode_err = np.linalg.norm(resid) / max(1e-16, np.linalg.norm(mode_rot))
    assert rel_mode_err < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the discretization and check that the numerical critical load approaches
    the analytical Euler value with decreasing relative error, and that the finest
    mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    L = 15.0
    r = 0.6
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    P_ref = 1.0
    P_cr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
    n_elems_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elem in n_elems_list:
        n_nodes = n_elem + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(n_elem):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        P_cr_num = lam * P_ref
        rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * 1.05 + 1e-12
    assert errors[-1] < 1e-05