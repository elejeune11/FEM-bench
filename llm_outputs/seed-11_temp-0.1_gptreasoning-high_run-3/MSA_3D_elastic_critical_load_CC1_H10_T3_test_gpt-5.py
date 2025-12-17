def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical value.
    Use 10 elements and verify relative error is within a reasonable tolerance.
    """
    E = 210000000000.0
    nu = 0.3
    ne = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for r in radii:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in lengths:
            n_nodes = ne + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack((np.zeros(n_nodes), np.zeros(n_nodes), z))
            elements = []
            for i in range(ne):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert lam > 0.0
            P_cr_num = lam
            K_eff = 2.0
            P_cr_euler = math.pi ** 2 * E * I / (K_eff * L) ** 2
            rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
            assert rel_err < 0.02

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load.
    The critical load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R,
    allowing for arbitrary scale and sign.
    """
    E = 210000000000.0
    nu = 0.3
    ne = 10
    L = 10.0
    b = 0.2
    h = 0.1
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = I_y + I_z
    n_nodes = ne + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.column_stack((np.zeros(n_nodes), np.zeros(n_nodes), z))
    elements_base = []
    for i in range(ne):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    lam_base, phi_base = fcn(node_coords, elements_base, boundary_conditions, nodal_loads_base)
    axis = np.array([1.0, 2.0, 3.0], dtype=float)
    axis /= np.linalg.norm(axis)
    theta = 0.7
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    local_z_base = np.array([0.0, 1.0, 0.0], dtype=float)
    local_z_rot = (R @ local_z_base).tolist()
    for i in range(ne):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    tip_load_vec_base = np.array([0.0, 0.0, -1.0], dtype=float)
    tip_load_vec_rot = (R @ tip_load_vec_base).tolist()
    nodal_loads_rot = {n_nodes - 1: [tip_load_vec_rot[0], tip_load_vec_rot[1], tip_load_vec_rot[2], 0.0, 0.0, 0.0]}
    lam_rot, phi_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lam_base > 0.0 and lam_rot > 0.0
    rel_diff_lam = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lam < 1e-06
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    v_target = T @ phi_base
    denom = float(v_target @ v_target)
    if denom == 0.0:
        s = 1.0
    else:
        s = float(phi_rot @ v_target) / denom
    num = np.linalg.norm(phi_rot - s * v_target)
    den = np.linalg.norm(phi_rot) if np.linalg.norm(phi_rot) > 0 else 1.0
    mode_mismatch = num / den
    assert mode_mismatch < 1e-05

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    L = 10.0
    r = 0.3
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    ne_list = [2, 4, 8, 16, 32, 64]
    errors = []
    for ne in ne_list:
        n_nodes = ne + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack((np.zeros(n_nodes), np.zeros(n_nodes), z))
        elements = []
        for i in range(ne):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_num = lam
        K_eff = 2.0
        P_cr_euler = math.pi ** 2 * E * I / (K_eff * L) ** 2
        rel_err = abs(P_cr_num - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * (1.0 + 0.001)
    assert errors[-1] < 0.0001