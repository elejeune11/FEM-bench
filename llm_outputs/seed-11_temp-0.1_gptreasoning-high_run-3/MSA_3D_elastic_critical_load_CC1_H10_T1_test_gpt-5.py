def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever circular column Euler buckling: parameter sweep over radii and lengths using 10 elements."""
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    ne = 10
    P_ref_mag = 1.0
    rtol = 5e-05
    for r in radii:
        A = np.pi * r * r
        I = np.pi * r ** 4 / 4.0
        J = 2.0 * I
        for L in lengths:
            n_nodes = ne + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = z
            elements = []
            for i in range(ne):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([1.0, 0.0, 0.0])})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref_mag, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert np.isfinite(lam) and lam > 0.0
            assert isinstance(mode, np.ndarray) and mode.shape == (6 * n_nodes,)
            P_cr_num = lam * P_ref_mag
            P_cr_euler = np.pi ** 2 * E * I / (4.0 * L * L)
            rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_err < rtol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance with rectangular section: eigenvalue invariant under rigid rotation and mode transforms with T."""
    E = 200000000000.0
    nu = 0.3
    L = 12.0
    ne = 12
    n_nodes = ne + 1
    A = 0.01
    I_y = 1e-05
    I_z = 4e-05
    J = I_y + I_z
    P_ref_mag = 1.0
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.zeros((n_nodes, 3))
    node_coords[:, 2] = z
    elements = []
    for i in range(ne):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([1.0, 0.0, 0.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref_mag, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.isfinite(lam_base) and lam_base > 0.0
    assert isinstance(mode_base, np.ndarray) and mode_base.shape == (6 * n_nodes,)
    yaw, pitch, roll = (0.3, -0.5, 0.2)
    cz, sz = (np.cos(yaw), np.sin(yaw))
    cy, sy = (np.cos(pitch), np.sin(pitch))
    cx, sx = (np.cos(roll), np.sin(roll))
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    R = Rz @ Ry @ Rx
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        local_z_rot = R @ np.array(e['local_z'])
        elements_rot.append({'node_i': e['node_i'], 'node_j': e['node_j'], 'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'local_z': local_z_rot})
    F_base = np.array([0.0, 0.0, -P_ref_mag])
    F_rot = R @ F_base
    nodal_loads_rot = {n_nodes - 1: [F_rot[0], F_rot[1], F_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isfinite(lam_rot) and lam_rot > 0.0
    assert isinstance(mode_rot, np.ndarray) and mode_rot.shape == (6 * n_nodes,)
    assert abs(lam_rot - lam_base) / lam_base < 1e-09
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    x = T @ mode_base
    y = mode_rot
    alpha = float(y @ x) / float(x @ x)
    rel_diff = np.linalg.norm(y - alpha * x) / np.linalg.norm(y)
    assert rel_diff < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Mesh convergence for Euler buckling of a fixed–free circular cantilever; error decreases with refinement and finest is very accurate."""
    E = 210000000000.0
    nu = 0.3
    L = 20.0
    r = 0.5
    A = np.pi * r * r
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    P_ref_mag = 1.0
    ne_list = [8, 16, 32, 64]
    errors = []
    P_cr_euler = np.pi ** 2 * E * I / (4.0 * L * L)
    for ne in ne_list:
        n_nodes = ne + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = z
        elements = []
        for i in range(ne):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([1.0, 0.0, 0.0])})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref_mag, 0.0, 0.0, 0.0]}
        lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert np.isfinite(lam) and lam > 0.0
        assert isinstance(mode, np.ndarray) and mode.shape == (6 * n_nodes,)
        P_cr_num = lam * P_ref_mag
        rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    diffs = np.diff(errors)
    assert np.all(diffs < 0.0)
    assert errors[-1] < 5e-06