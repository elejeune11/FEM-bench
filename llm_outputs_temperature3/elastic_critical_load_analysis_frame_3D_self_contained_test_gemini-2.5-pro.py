def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    P_ref = 1.0
    n_elements = 10
    n_nodes = n_elements + 1
    for r in [0.5, 0.75, 1.0]:
        for L in [10, 20, 40]:
            A = np.pi * r ** 2
            Iy = np.pi * r ** 4 / 4
            Iz = Iy
            J = 2 * Iy
            I_rho = Iy + Iz
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': None})
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lambda_cr, _) = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
            P_cr_analytical = np.pi ** 2 * E * Iy / (2 * L) ** 2
            P_cr_numerical = lambda_cr * P_ref
            assert P_cr_numerical == pytest.approx(P_cr_analytical, rel=0.0001)

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
    n_elements = 8
    n_nodes = n_elements + 1
    E = 1.0
    nu = 0.3
    P_ref = 1.0
    (b, h) = (0.5, 1.0)
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = 0.141 * b ** 3 * h
    I_rho = Iy + Iz
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_nodes)
    local_z_base = np.array([0.0, 1.0, 0.0])
    elements_base = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_base} for i in range(n_elements)]
    bcs = {0: [True] * 6}
    load_vec_base = np.array([0.0, 0.0, -P_ref, 0.0, 0.0, 0.0])
    loads_base = {n_nodes - 1: load_vec_base}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, bcs, loads_base)
    R = Rotation.from_euler('xyz', [20, -30, 45], degrees=True).as_matrix()
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ local_z_base
    elements_rot = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot} for i in range(n_elements)]
    F_rot = R @ load_vec_base[:3]
    M_rot = R @ load_vec_base[3:]
    load_vec_rot = np.concatenate([F_rot, M_rot])
    loads_rot = {n_nodes - 1: load_vec_rot}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bcs, loads_rot)
    assert lambda_rot == pytest.approx(lambda_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        start = 6 * i
        T[start:start + 3, start:start + 3] = R
        T[start + 3:start + 6, start + 3:start + 6] = R
    mode_base_transformed = T @ mode_base
    norm_rot = np.linalg.norm(mode_rot)
    norm_base_t = np.linalg.norm(mode_base_transformed)
    assert norm_rot > 1e-09, 'Rotated mode shape is near zero.'
    assert norm_base_t > 1e-09, 'Transformed base mode shape is near zero.'
    cosine_similarity = np.dot(mode_rot, mode_base_transformed) / (norm_rot * norm_base_t)
    assert abs(cosine_similarity) == pytest.approx(1.0)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 10.0
    r = 0.5
    E = 1.0
    nu = 0.3
    P_ref = 1.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    I_rho = 2 * I
    P_cr_analytical = np.pi ** 2 * E * I / (2 * L) ** 2
    mesh_sizes = [2, 4, 8, 16]
    errors = []
    for n_elements in mesh_sizes:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1, 0, 0]} for i in range(n_elements)]
        boundary_conditions = {0: [True] * 6}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_numerical = lambda_cr * P_ref
        relative_error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
        errors.append(relative_error)
    assert errors[0] > errors[1] > errors[2] > errors[3]
    rate1 = errors[0] / errors[1]
    rate2 = errors[1] / errors[2]
    rate3 = errors[2] / errors[3]
    assert rate1 == pytest.approx(4.0, rel=0.1)
    assert rate2 == pytest.approx(4.0, rel=0.1)
    assert rate3 == pytest.approx(4.0, rel=0.1)
    assert errors[-1] < 1e-05