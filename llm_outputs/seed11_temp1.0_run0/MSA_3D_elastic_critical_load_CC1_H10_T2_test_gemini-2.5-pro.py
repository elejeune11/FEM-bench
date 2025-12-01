def test_euler_buckling_cantilever_circular_param_sweep(fcn: Callable):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler
    cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated
    discretization error at 10-5.
    """
    E = 200.0
    nu = 0.3
    n_elements = 10
    n_nodes = n_elements + 1
    P_ref = -1.0
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    for r in radii:
        for L in lengths:
            I = np.pi * r ** 4 / 4
            p_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            A = np.pi * r ** 2
            J = 2 * I
            elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J} for i in range(n_elements)]
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0, 0, P_ref, 0, 0, 0]}
            (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            p_cr_numerical = lambda_cr * abs(P_ref)
            assert p_cr_numerical == pytest.approx(p_cr_analytical, rel=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn: Callable):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    (L, b, h) = (20.0, 2.0, 1.0)
    (E, nu) = (200.0, 0.3)
    n_elements = 8
    n_nodes = n_elements + 1
    P_ref = -1.0
    A = b * h
    I_y = h * b ** 3 / 12
    I_z = b * h ** 3 / 12
    J = I_y + I_z
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_nodes)
    local_z_base = np.array([0.0, 1.0, 0.0])
    elements_base = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_base} for i in range(n_elements)]
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    loads_base = {n_nodes - 1: [0, 0, P_ref, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, bcs, loads_base)
    rot_axis = np.array([1, -2, 0.5])
    rot_axis /= np.linalg.norm(rot_axis)
    rot_angle = np.deg2rad(35)
    R = Rotation.from_rotvec(rot_angle * rot_axis)
    R_mat = R.as_matrix()
    node_coords_rot = (R_mat @ node_coords_base.T).T
    local_z_rot = R_mat @ local_z_base
    elements_rot = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot} for i in range(n_elements)]
    F_base = np.array([0, 0, P_ref])
    M_base = np.array([0, 0, 0])
    F_rot = R_mat @ F_base
    M_rot = R_mat @ M_base
    loads_rot = {n_nodes - 1: np.concatenate((F_rot, M_rot))}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bcs, loads_rot)
    assert lambda_rot == pytest.approx(lambda_base)
    T_block = np.block([[R_mat, np.zeros((3, 3))], [np.zeros((3, 3)), R_mat]])
    T = np.kron(np.eye(n_nodes), T_block)
    mode_base_transformed = T @ mode_base
    norm_rot = np.linalg.norm(mode_rot)
    norm_base_t = np.linalg.norm(mode_base_transformed)
    assert norm_rot > 1e-09 and norm_base_t > 1e-09
    u_rot = mode_rot / norm_rot
    u_base_t = mode_base_transformed / norm_base_t
    assert abs(np.dot(u_rot, u_base_t)) == pytest.approx(1.0)

def test_cantilever_euler_buckling_mesh_convergence(fcn: Callable):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    (L, r) = (20.0, 0.5)
    (E, nu) = (200.0, 0.3)
    P_ref = -1.0
    I = np.pi * r ** 4 / 4
    p_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    n_elements_list = [2, 4, 8, 16, 32]
    errors = []
    A = np.pi * r ** 2
    J = 2 * I
    for n_elements in n_elements_list:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J} for i in range(n_elements)]
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0, 0, P_ref, 0, 0, 0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        p_cr_numerical = lambda_cr * abs(P_ref)
        relative_error = abs(p_cr_numerical - p_cr_analytical) / p_cr_analytical
        errors.append(relative_error)
    for i in range(len(errors) - 1):
        assert errors[i] > errors[i + 1]
    assert errors[-1] < 1e-07