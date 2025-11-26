def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    n_elements = 10
    P_ref_val = -1.0
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    for r in radii:
        for L in lengths:
            I = np.pi * r ** 4 / 4.0
            p_crit_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
            node_coords = np.zeros((n_elements + 1, 3))
            node_coords[:, 2] = np.linspace(0, L, n_elements + 1)
            A = np.pi * r ** 2
            J = 2 * I
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_elements: [0, 0, P_ref_val, 0, 0, 0]}
            (lambda_crit, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            p_crit_numerical = lambda_crit * abs(P_ref_val)
            assert p_crit_numerical == pytest.approx(p_crit_analytical, rel=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    L = 20.0
    (b, h) = (0.1, 0.2)
    E = 210000000000.0
    nu = 0.3
    n_elements = 8
    P_ref_val = -1000.0
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    (a_j, b_j) = (max(h, b), min(h, b))
    J = a_j * b_j ** 3 * (1 / 3 - 0.21 * (b_j / a_j) * (1 - b_j ** 4 / (12 * a_j ** 4)))
    nodes_base = np.zeros((n_elements + 1, 3))
    nodes_base[:, 2] = np.linspace(0, L, n_elements + 1)
    local_z_base = np.array([1.0, 0.0, 0.0])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_base})
    bcs_base = {0: [1, 1, 1, 1, 1, 1]}
    load_vec_base = np.array([0, 0, P_ref_val, 0, 0, 0])
    loads_base = {n_elements: load_vec_base}
    (lambda_base, mode_base) = fcn(nodes_base, elements_base, bcs_base, loads_base)
    rot_vec = np.deg2rad(30) * np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3])
    R = ScipyRotation.from_rotvec(rot_vec).as_matrix()
    nodes_rot = (R @ nodes_base.T).T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    bcs_rot = bcs_base
    F_base = load_vec_base[:3]
    M_base = load_vec_base[3:]
    load_vec_rot = np.hstack([R @ F_base, R @ M_base])
    loads_rot = {n_elements: load_vec_rot}
    (lambda_rot, mode_rot) = fcn(nodes_rot, elements_rot, bcs_rot, loads_rot)
    assert lambda_rot == pytest.approx(lambda_base)
    n_nodes = n_elements + 1
    R_6x6_block = linalg.block_diag(R, R)
    T = linalg.block_diag(*[R_6x6_block] * n_nodes)
    mode_base_transformed = T @ mode_base
    norm_rot = mode_rot / np.linalg.norm(mode_rot)
    norm_base_transformed = mode_base_transformed / np.linalg.norm(mode_base_transformed)
    assert abs(np.dot(norm_rot, norm_base_transformed)) == pytest.approx(1.0)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 20.0
    r = 0.5
    E = 200000000000.0
    nu = 0.3
    P_ref_val = -1.0
    I = np.pi * r ** 4 / 4.0
    p_crit_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elements in n_elements_list:
        node_coords = np.zeros((n_elements + 1, 3))
        node_coords[:, 2] = np.linspace(0, L, n_elements + 1)
        A = np.pi * r ** 2
        J = 2 * I
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [0, 0, P_ref_val, 0, 0, 0]}
        (lambda_crit, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        p_crit_numerical = lambda_crit * abs(P_ref_val)
        error = abs(p_crit_numerical - p_crit_analytical) / p_crit_analytical
        errors.append(error)
    errors = np.array(errors)
    assert np.all(np.diff(errors) < 1e-12)
    assert errors[-1] < 1e-07