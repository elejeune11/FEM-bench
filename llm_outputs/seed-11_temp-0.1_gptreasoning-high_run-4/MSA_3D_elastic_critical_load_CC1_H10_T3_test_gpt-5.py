def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical solution.
    Use 10 elements and assert relative error consistent with high-accuracy discretization.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    tol = 0.0001
    max_rel_err = 0.0
    for r in radii:
        A = np.pi * r ** 2
        I = 0.25 * np.pi * r ** 4
        J = 0.5 * np.pi * r ** 4
        for L in lengths:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0], dtype=float)})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_true = np.pi ** 2 * E * I / (4.0 * L ** 2)
            P_cr_num = lam * 1.0
            rel_err = abs(P_cr_num - P_cr_true) / P_cr_true
            max_rel_err = max(max_rel_err, rel_err)
            assert np.isfinite(lam) and lam > 0.0
            assert mode.shape == (6 * n_nodes,)
    assert max_rel_err < tol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 210000000000.0
    nu = 0.3
    b = 0.15
    h = 0.25
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = I_y + I_z
    L = 12.0
    n_elems = 12
    n_nodes = n_elems + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0], dtype=float)})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords, elements, boundary_conditions, nodal_loads)

    def rot_x(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def rot_y(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

    def rot_z(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
    ax, ay, az = np.deg2rad([17.0, -10.0, 25.0])
    R = rot_z(az) @ rot_y(ay) @ rot_x(ax)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for e in elements:
        e_rot = dict(e)
        e_rot['local_z'] = (R @ np.array(e['local_z'], dtype=float)).astype(float)
        elements_rot.append(e_rot)
    tip_force_base = np.array([0.0, 0.0, -1.0], dtype=float)
    tip_force_rot = (R @ tip_force_base).astype(float)
    nodal_loads_rot = {n_nodes - 1: [tip_force_rot[0], tip_force_rot[1], tip_force_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isfinite(lam_base) and np.isfinite(lam_rot)
    assert lam_base > 0.0 and lam_rot > 0.0
    rel_diff_lam = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lam < 1e-08
    dof = 6 * n_nodes
    T = np.zeros((dof, dof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    v_pred = T @ mode_base
    alpha = float(v_pred @ mode_rot) / float(v_pred @ v_pred) if np.linalg.norm(v_pred) > 0 else 1.0
    res = mode_rot - alpha * v_pred
    rel_res = np.linalg.norm(res) / (np.linalg.norm(mode_rot) + 1e-16)
    assert rel_res < 5e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    r = 0.6
    A = np.pi * r ** 2
    I = 0.25 * np.pi * r ** 4
    J = 0.5 * np.pi * r ** 4
    L = 15.0
    n_elems_list = [2, 4, 8, 16, 32]
    errors = []
    P_cr_true = np.pi ** 2 * E * I / (4.0 * L ** 2)
    for n_elems in n_elems_list:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0], dtype=float)})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_num = lam * 1.0
        rel_err = abs(P_cr_num - P_cr_true) / P_cr_true
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]
    assert errors[-1] < 2e-05