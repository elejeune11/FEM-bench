def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical value.
    Use 10 elements; assert relative error within a modest tolerance accounting for discretization.
    """

    def euler_cantilever(E, I, L):
        return math.pi ** 2 * E * I / (2.0 * L) ** 2
    E = 200000.0
    nu = 0.3
    n_el = 10
    axis_dir = np.array([0.0, 0.0, 1.0])
    local_z = np.array([1.0, 0.0, 0.0])
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    tol = 0.05
    for r in radii:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in lengths:
            n_nodes = n_el + 1
            zs = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack((np.zeros_like(zs), np.zeros_like(zs), zs))
            elements = []
            for i in range(n_el):
                elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I, I_z=I, J=J, local_z=local_z))
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            F_ref = -1.0 * axis_dir
            nodal_loads = {n_nodes - 1: [F_ref[0], F_ref[1], F_ref[2], 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            p_euler = euler_cantilever(E, I, L)
            rel_err = abs(lam - p_euler) / p_euler
            assert lam > 0.0
            assert mode is not None and np.size(mode) == 6 * n_nodes
            assert rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever in its original orientation and again after applying a rigid-body rotation R
    to geometry, element axes, and applied load. λ should be identical.
    The buckling mode from the rotated model should equal the base mode transformed by a
    block-diagonal T built from R for both translational and rotational DOFs.
    """

    def rot_x(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    E = 250000.0
    nu = 0.29
    L = 12.0
    n_el = 12
    n_nodes = n_el + 1
    A = 3.0
    Iy = 2.0
    Iz = 0.5
    J = Iy + Iz
    axis_dir = np.array([0.0, 0.0, 1.0])
    local_z_base = np.array([1.0, 0.0, 0.0])
    zs = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.column_stack((np.zeros_like(zs), np.zeros_like(zs), zs))
    elements_base = []
    for i in range(n_el):
        elements_base.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=local_z_base))
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_ref_base = -1.0 * axis_dir
    nodal_loads_base = {n_nodes - 1: [F_ref_base[0], F_ref_base[1], F_ref_base[2], 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    mode_base = np.asarray(mode_base).reshape(-1)
    R = rot_z(0.3) @ rot_y(0.4) @ rot_x(0.5)
    node_coords_rot = node_coords_base @ R.T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for i in range(n_el):
        elements_rot.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=local_z_rot))
    axis_dir_rot = R @ axis_dir
    F_ref_rot = -1.0 * axis_dir_rot
    nodal_loads_rot = {n_nodes - 1: [F_ref_rot[0], F_ref_rot[1], F_ref_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    mode_rot = np.asarray(mode_rot).reshape(-1)
    assert lam_base > 0.0 and lam_rot > 0.0
    rel_diff_lam = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lam < 1e-06
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_pred = T @ mode_base
    denom = float(np.dot(mode_pred, mode_pred))
    if denom == 0.0:
        alpha = 1.0
    else:
        alpha = float(np.dot(mode_pred, mode_rot) / denom)
    diff = mode_rot - alpha * mode_pred
    rel_err_mode = np.linalg.norm(diff) / max(np.linalg.norm(mode_rot), 1e-12)
    assert rel_err_mode < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check numerical critical load approaches the analytical Euler value,
    with decreasing relative error and very high accuracy on the finest mesh.
    """

    def euler_cantilever(E, I, L):
        return math.pi ** 2 * E * I / (2.0 * L) ** 2
    E = 200000.0
    nu = 0.3
    r = 0.7
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    L = 15.0
    axis_dir = np.array([0.0, 0.0, 1.0])
    local_z = np.array([1.0, 0.0, 0.0])
    p_euler = euler_cantilever(E, I, L)
    n_el_list = [2, 4, 8, 16]
    errors = []
    for n_el in n_el_list:
        n_nodes = n_el + 1
        zs = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack((np.zeros_like(zs), np.zeros_like(zs), zs))
        elements = []
        for i in range(n_el):
            elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I, I_z=I, J=J, local_z=local_z))
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        F_ref = -1.0 * axis_dir
        nodal_loads = {n_nodes - 1: [F_ref[0], F_ref[1], F_ref[2], 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_err = abs(lam - p_euler) / p_euler
        errors.append(rel_err)
    assert all((e > 0 for e in errors))
    assert errors[-1] < errors[-2] < errors[-3] < errors[0]
    assert errors[-1] < 0.01