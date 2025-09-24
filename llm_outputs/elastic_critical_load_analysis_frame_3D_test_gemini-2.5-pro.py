def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 210000000000.0
    nu = 0.3
    n_elements = 10
    n_nodes = n_elements + 1
    P_ref = 1.0
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    for r in radii:
        for L in lengths:
            I = np.pi * r ** 4 / 4
            P_cr_analytical = np.pi ** 2 * E * I / (2.0 * L) ** 2
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'Iy': I, 'Iz': I, 'J': np.pi * r ** 4 / 2, 'I_rho': np.pi * r ** 4 / 2, 'local_z': [1.0, 0.0, 0.0]})
            boundary_conditions = {0: [True] * 6}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_numerical = lambda_cr * P_ref
            assert P_cr_numerical == pytest.approx(P_cr_analytical, rel=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    L = 15.0
    n_elements = 8
    n_nodes = n_elements + 1
    E = 200000000000.0
    nu = 0.3
    (b, h) = (0.1, 0.2)
    P_ref = 1.0
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    I_rho = Iy + Iz
    (a_j, b_j) = (max(b, h), min(b, h))
    J = a_j * b_j ** 3 * (1 / 3 - 0.21 * (b_j / a_j) * (1 - b_j ** 4 / (12 * a_j ** 4)))
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_nodes)
    local_z_base = np.array([1.0, 0.0, 0.0])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_base})
    bc_base = {0: [True] * 6}
    load_vec_base = np.array([0.0, 0.0, -P_ref, 0.0, 0.0, 0.0])
    loads_base = {n_nodes - 1: load_vec_base}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, bc_base, loads_base)
    R = Rotation.from_euler('xyz', [20, -30, 50], degrees=True).as_matrix()
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for elem in elements_base:
        elem_rot = elem.copy()
        elem_rot['local_z'] = local_z_rot
        elements_rot.append(elem_rot)
    bc_rot = bc_base
    F_rot = R @ load_vec_base[:3]
    M_rot = R @ load_vec_base[3:]
    loads_rot = {n_nodes - 1: np.hstack([F_rot, M_rot])}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc_rot, loads_rot)
    assert lambda_rot == pytest.approx(lambda_base)
    T_node = block_diag(R, R)
    T_global = block_diag(*[T_node] * n_nodes)
    mode_base_transformed = T_global @ mode_base
    norm_rot = np.linalg.norm(mode_rot)
    norm_base_transformed = np.linalg.norm(mode_base_transformed)
    assert norm_rot > 1e-09 and norm_base_transformed > 1e-09
    cosine_similarity = np.dot(mode_rot, mode_base_transformed) / (norm_rot * norm_base_transformed)
    assert abs(cosine_similarity) == pytest.approx(1.0)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 20.0
    r = 0.5
    E = 210000000000.0
    nu = 0.3
    P_ref = 1.0
    I = np.pi * r ** 4 / 4
    P_cr_analytical = np.pi ** 2 * E * I / (2.0 * L) ** 2
    mesh_sizes = [2, 4, 8, 16, 32]
    errors = []
    for n_elements in mesh_sizes:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'Iy': I, 'Iz': I, 'J': np.pi * r ** 4 / 2, 'I_rho': np.pi * r ** 4 / 2, 'local_z': [1.0, 0.0, 0.0]})
        boundary_conditions = {0: [True] * 6}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_numerical = lambda_cr * P_ref
        rel_error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
        errors.append(rel_error)
    assert np.all(np.diff(errors) < 0), 'Error is not monotonically decreasing'
    assert errors[-1] < 1e-07, 'Finest mesh did not achieve required accuracy'