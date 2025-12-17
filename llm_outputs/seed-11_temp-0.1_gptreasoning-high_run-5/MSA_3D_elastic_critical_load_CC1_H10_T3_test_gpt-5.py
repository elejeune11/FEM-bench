def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements and a tight tolerance compatible with discretization accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_el = 10
    tol_rtol = 0.002
    for r in radii:
        A = math.pi * r * r
        I = 0.25 * math.pi * r ** 4
        J = 2.0 * I
        for L in lengths:
            n_nodes = n_el + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
            elements = []
            for i in range(n_el):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            lam, mode = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            assert np.isfinite(lam) and lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            P_cr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam - P_cr_euler) / P_cr_euler
            assert rel_err < tol_rtol

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 200000000000.0
    nu = 0.3
    b_z = 0.25
    h_y = 0.1
    A = b_z * h_y
    I_y = b_z ** 3 * h_y / 12.0
    I_z = b_z * h_y ** 3 / 12.0
    J = b_z * h_y ** 3 / 3.0
    L = 15.0
    n_el = 12
    n_nodes = n_el + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords_base = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
    elements_base = []
    for i in range(n_el):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_base = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    lam_base, mode_base = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    assert np.isfinite(lam_base) and lam_base > 0.0
    assert mode_base.shape == (6 * n_nodes,)
    axis = np.array([0.35, -0.41, 0.84], dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = 0.63
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = (R @ np.array([0.0, 1.0, 0.0])).tolist()
    elements_rot = []
    for i in range(n_el):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    load_vec_rot = -(R @ np.array([0.0, 0.0, 1.0]))
    nodal_loads_rot = {n_nodes - 1: [float(load_vec_rot[0]), float(load_vec_rot[1]), float(load_vec_rot[2]), 0.0, 0.0, 0.0]}
    lam_rot, mode_rot = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isfinite(lam_rot) and lam_rot > 0.0
    assert mode_rot.shape == (6 * n_nodes,)
    rel_diff_lam = abs(lam_rot - lam_base) / lam_base
    assert rel_diff_lam < 1e-08
    T_blocks = []
    for _ in range(n_nodes):
        T_node = np.zeros((6, 6))
        T_node[0:3, 0:3] = R
        T_node[3:6, 3:6] = R
        T_blocks.append(T_node)
    T = np.block([[T_blocks[i] if i == j else np.zeros((6, 6)) for j in range(n_nodes)] for i in range(n_nodes)])
    v_base_map = T @ mode_base
    denom = float(np.dot(v_base_map, v_base_map)) + 0.0
    assert denom > 0.0
    alpha = float(np.dot(v_base_map, mode_rot)) / denom
    resid = np.linalg.norm(mode_rot - alpha * v_base_map) / np.linalg.norm(mode_rot)
    assert resid < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.75
    A = math.pi * r ** 2
    I = 0.25 * math.pi * r ** 4
    J = 2.0 * I
    L = 30.0
    P_cr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
    n_list = [2, 4, 8, 16, 32]
    errors = []
    for n_el in n_list:
        n_nodes = n_el + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.column_stack([np.zeros(n_nodes), np.zeros(n_nodes), z])
        elements = []
        for i in range(n_el):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 1.0, 0.0]})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        lam, _ = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert np.isfinite(lam) and lam > 0.0
        rel_err = abs(lam - P_cr_euler) / P_cr_euler
        errors.append(rel_err)
    assert errors[0] > errors[-1]
    improvements = sum((1 for k in range(1, len(errors)) if errors[k] < errors[k - 1]))
    assert improvements >= 3
    assert errors[-1] < 0.0001