def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z. Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements; enforce tight relative tolerance consistent with expected discretization accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    tol_rel = 0.002
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_el = 10
    for (r, L) in product(radii, lengths):
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        I_rho = J
        z_coords = np.linspace(0.0, L, n_el + 1)
        node_coords = np.column_stack([np.zeros_like(z_coords), np.zeros_like(z_coords), z_coords])
        elements = []
        for e in range(n_el):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_el: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert isinstance(lam, float)
        assert mode.shape == (6 * (n_el + 1),)
        assert lam > 0.0
        P_cr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(lam - P_cr_euler) / P_cr_euler
        assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever model, then apply a rigid-body rotation R to geometry, element axes, and load.
    The critical load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R (up to scale/sign),
    where T is block-diagonal with R for both translational and rotational DOFs at each node.
    """

    def rotation_matrix_from_axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        (x, y, z) = axis
        c = math.cos(angle)
        s = math.sin(angle)
        C = 1.0 - c
        return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], dtype=float)
    E = 200000000000.0
    nu = 0.3
    b = 0.3
    h = 0.5
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = Iy + Iz
    I_rho = J
    L = 12.0
    n_el = 12
    z_coords = np.linspace(0.0, L, n_el + 1)
    node_coords = np.column_stack([np.zeros_like(z_coords), np.zeros_like(z_coords), z_coords])
    elements = []
    base_local_z = np.array([1.0, 0.0, 0.0])
    for e in range(n_el):
        elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': base_local_z.tolist()})
    boundary_conditions = {0: [True, True, True, True, True, True]}
    nodal_loads = {n_el: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert mode_base.shape == (6 * (n_el + 1),)
    assert lam_base > 0.0
    axis = [1.0, 2.0, 3.0]
    angle = 0.7
    R = rotation_matrix_from_axis_angle(axis, angle)
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ base_local_z
    elements_rot = []
    for e in range(n_el):
        elements_rot.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot.tolist()})
    F_tip = np.array([0.0, 0.0, -1.0])
    F_tip_rot = R @ F_tip
    nodal_loads_rot = {n_el: [F_tip_rot[0], F_tip_rot[1], F_tip_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert mode_rot.shape == (6 * (n_el + 1),)
    assert lam_rot > 0.0
    rel_diff = abs(lam_rot - lam_base) / lam_base
    assert rel_diff < 1e-08
    n_nodes = n_el + 1
    T_blocks = []
    for _ in range(n_nodes):
        T_node = np.zeros((6, 6), dtype=float)
        T_node[0:3, 0:3] = R
        T_node[3:6, 3:6] = R
        T_blocks.append(T_node)
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = T_blocks[i]
    mode_base_transformed = T @ mode_base
    denom = float(mode_base_transformed @ mode_base_transformed)
    assert denom > 0.0
    alpha = float(mode_rot @ mode_base_transformed) / denom
    residual = mode_rot - alpha * mode_base_transformed
    res_rel = np.linalg.norm(residual) / np.linalg.norm(mode_rot)
    assert res_rel < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load approaches
    the analytical Euler value with decreasing relative error; finest mesh achieves high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 1.0
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    I_rho = J
    L = 20.0
    meshes = [4, 8, 16, 32, 64]
    errors = []
    last_mode_shape_len = None
    for n_el in meshes:
        z_coords = np.linspace(0.0, L, n_el + 1)
        node_coords = np.column_stack([np.zeros_like(z_coords), np.zeros_like(z_coords), z_coords])
        elements = []
        for e in range(n_el):
            elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_el: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert lam > 0.0
        assert mode.shape == (6 * (n_el + 1),)
        last_mode_shape_len = mode.shape[0]
        P_cr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(lam - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    assert last_mode_shape_len == 6 * (meshes[-1] + 1)
    assert errors[0] > errors[-1]
    num_decreases = sum((1 for i in range(1, len(errors)) if errors[i] < errors[i - 1]))
    assert num_decreases >= len(errors) - 2
    assert errors[-1] < 0.001