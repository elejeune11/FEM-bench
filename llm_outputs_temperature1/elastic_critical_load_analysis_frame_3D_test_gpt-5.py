def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z. Sweep through radii r ∈ {0.5, 0.75, 1.0}
    and lengths L ∈ {10, 20, 40}. For each case, run the full pipeline and compare λ·P_ref to the
    Euler cantilever analytical solution. Use 10 elements and tight tolerance suitable for this mesh.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    ne = 10
    P_ref = 1.0
    tol = 0.0001
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        I_rho = J
        for L in lengths:
            z = np.linspace(0.0, L, ne + 1)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for e in range(ne):
                elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {ne: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_num = lam * P_ref
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(P_num - P_euler) / P_euler
            assert rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever model, then rotate geometry, local axes, and load by a rigid-body rotation R.
    The critical load factor λ should be identical. The buckling mode from the rotated model should
    equal the base mode transformed by T = blockdiag(R, R) per node, up to a scalar.
    """
    E = 210000000000.0
    nu = 0.3
    b = 0.2
    h = 0.1
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    L = 12.0
    ne = 12
    z = np.linspace(0.0, L, ne + 1)
    node_coords_base = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements_base = []
    for e in range(ne):
        elements_base.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
    boundary_conditions = {0: [True, True, True, True, True, True]}
    P_ref = 1.0
    nodal_loads_base = {ne: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)

    def rot_x(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    R = rot_z(0.23) @ rot_y(-0.51) @ rot_x(0.37)
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_base = np.array([1.0, 0.0, 0.0])
    local_z_rot = (R @ local_z_base).tolist()
    elements_rot = []
    for e in range(ne):
        elements_rot.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -P_ref])
    F_rot = R @ F_base
    nodal_loads_rot = {ne: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff = abs(lam_rot - lam_base) / abs(lam_base)
    assert rel_diff < 1e-08
    n_nodes = node_coords_base.shape[0]
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_pred_rot = T @ mode_base
    denom = np.dot(mode_pred_rot, mode_pred_rot)
    if denom == 0.0:
        alpha = 1.0
    else:
        alpha = float(np.dot(mode_pred_rot, mode_rot) / denom)
    residual = np.linalg.norm(mode_rot - alpha * mode_pred_rot) / (np.linalg.norm(mode_rot) + 1e-30)
    assert residual < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load approaches
    the analytical Euler value with decreasing relative error, and that the finest mesh
    achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.3
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    I_rho = J
    L = 18.0
    meshes = [2, 4, 8, 16, 32]
    P_ref = 1.0
    P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    errors = []
    for ne in meshes:
        z = np.linspace(0.0, L, ne + 1)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for e in range(ne):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {ne: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_num = lam * P_ref
        rel_err = abs(P_num - P_euler) / P_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-12
    assert errors[-1] < 1e-05