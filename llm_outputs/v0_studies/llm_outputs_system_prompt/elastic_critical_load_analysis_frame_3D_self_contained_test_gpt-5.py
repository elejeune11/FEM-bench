def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements and assert relative error is small, appropriate for discretization accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    P_ref = 1.0
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_el = 10
    for r in radii:
        A = pi * r ** 2
        Iy = Iz = pi * r ** 4 / 4.0
        J = pi * r ** 4 / 2.0
        I_rho = Iy + Iz
        for L in lengths:
            n_nodes = n_el + 1
            z_coords = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3), dtype=float)
            node_coords[:, 2] = z_coords
            elements = []
            for i in range(n_el):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0.0, 1.0, 0.0]})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert np.isfinite(lam)
            assert mode.shape == (6 * n_nodes,)
            P_euler = pi ** 2 * E * Iy / (4.0 * L ** 2)
            rel_err = abs(lam - P_euler) / P_euler
            assert rel_err < 0.0001

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever model in its base orientation, then rotate the geometry,
    element axes, and applied load by a rigid-body rotation R. The critical load
    factors should match, and the rotated mode should equal T @ base_mode up to scale/sign,
    where T applies R to both translational and rotational DOFs at each node.
    """
    E = 70000000000.0
    nu = 0.3
    b = 1.0
    h = 2.0
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = b * h ** 3 / 3.0
    I_rho = Iy + Iz
    L = 10.0
    n_el = 12
    n_nodes = n_el + 1
    z_coords = np.linspace(0.0, L, n_nodes)
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    node_coords[:, 2] = z_coords
    elements = []
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0.0, 1.0, 0.0]})
    boundary_conditions = {0: [True, True, True, True, True, True]}
    P_ref = 1.0
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.isfinite(lam_base)
    assert mode_base.shape == (6 * n_nodes,)
    ax = 0.37
    ay = -0.52
    az = 0.41
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]], dtype=float)
    Ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]], dtype=float)
    Rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]], dtype=float)
    R = Rz @ Ry @ Rx
    node_coords_rot = (R @ node_coords.T).T
    local_z_base = np.array([0.0, 1.0, 0.0], dtype=float)
    local_z_rot = (R @ local_z_base).tolist()
    elements_rot = []
    for i in range(n_el):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -P_ref], dtype=float)
    F_rot = (R @ F_base).tolist()
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isfinite(lam_rot)
    assert mode_rot.shape == (6 * n_nodes,)
    assert abs(lam_rot - lam_base) / abs(lam_base) < 1e-09
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    v = T @ mode_base
    nv = np.linalg.norm(v)
    nr = np.linalg.norm(mode_rot)
    if nv == 0.0 or nr == 0.0:
        assert False
    v /= nv
    w = mode_rot / nr
    d1 = np.linalg.norm(w - v)
    d2 = np.linalg.norm(w + v)
    d = min(d1, d2)
    assert d < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check the numerical critical load approaches
    the analytical Euler value with decreasing relative error, and that the finest
    mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    r = 1.0
    L = 20.0
    A = pi * r ** 2
    Iy = Iz = pi * r ** 4 / 4.0
    J = pi * r ** 4 / 2.0
    I_rho = Iy + Iz
    P_euler = pi ** 2 * E * Iy / (4.0 * L ** 2)
    meshes = [2, 4, 8, 16, 32]
    errors = []
    for n_el in meshes:
        n_nodes = n_el + 1
        z_coords = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3), dtype=float)
        node_coords[:, 2] = z_coords
        elements = []
        for i in range(n_el):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert np.isfinite(lam)
        errors.append(abs(lam - P_euler) / P_euler)
    errors = np.array(errors, dtype=float)
    assert np.all(np.diff(errors) < 0.0)
    assert errors[-1] < 1e-06