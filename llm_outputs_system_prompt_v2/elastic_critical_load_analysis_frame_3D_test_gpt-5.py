def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements and assert relative error within a modest tolerance suitable for this discretization.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    tol_rel = 0.02
    for r in radii:
        A = np.pi * r * r
        I = 0.25 * np.pi * r ** 4
        J = 0.5 * np.pi * r ** 4
        I_rho = J
        for L in lengths:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(n_elems):
                ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]}
                elements.append(ele)
            boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_euler = np.pi ** 2 * E * I / (4.0 * L * L)
            rel_err = abs(lam - P_euler) / P_euler
            assert rel_err < tol_rel

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance with a rectangular section (Iy ≠ Iz).
    Solve the cantilever model and again after rotating geometry, element axes, and load by R.
    The critical load factor λ should match, and the rotated mode should equal T @ mode_base up to scale.
    """
    E = 210000000000.0
    nu = 0.3
    dy = 0.1
    dz = 0.2
    A = dy * dz
    Iy = dz * dy ** 3 / 12.0
    Iz = dy * dz ** 3 / 12.0
    J = Iy + Iz
    I_rho = Iy + Iz
    L = 12.0
    n_elems = 12
    n_nodes = n_elems + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
    boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    ang_x = np.deg2rad(20.0)
    ang_y = np.deg2rad(-15.0)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ang_x), -np.sin(ang_x)], [0.0, np.sin(ang_x), np.cos(ang_x)]])
    Ry = np.array([[np.cos(ang_y), 0.0, np.sin(ang_y)], [0.0, 1.0, 0.0], [-np.sin(ang_y), 0.0, np.cos(ang_y)]])
    R = Ry @ Rx
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = (R @ np.array([1.0, 0.0, 0.0]).reshape(3, 1)).ravel()
    elements_rot = []
    for i in range(n_elems):
        e = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot.tolist()}
        elements_rot.append(e)
    F_base = np.array([0.0, 0.0, -1.0])
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_diff_lam = abs(lam_rot - lam_base) / max(abs(lam_base), 1.0)
    assert rel_diff_lam < 1e-06
    R6 = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = np.kron(np.eye(n_nodes), R6)
    mode_rot_pred = T @ mode_base
    denom = float(mode_rot_pred @ mode_rot_pred)
    assert denom > 0.0
    alpha = float(mode_rot @ mode_rot_pred) / denom
    resid = mode_rot - alpha * mode_rot_pred
    rel_resid = np.linalg.norm(resid) / max(np.linalg.norm(mode_rot), 1e-16)
    assert rel_resid < 5e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the mesh and verify that the numerical critical load approaches the analytical value with decreasing error.
    """
    E = 210000000000.0
    nu = 0.3
    r = 1.0
    A = np.pi * r * r
    I = 0.25 * np.pi * r ** 4
    J = 0.5 * np.pi * r ** 4
    I_rho = J
    L = 20.0
    P_euler = np.pi ** 2 * E * I / (4.0 * L * L)
    elem_counts = [4, 8, 16, 32]
    errors = []
    for n_elems in elem_counts:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(n_elems):
            ele = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]}
            elements.append(ele)
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        errors.append(abs(lam - P_euler) / P_euler)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]
    assert errors[-1] < 0.005