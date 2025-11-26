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
    P_ref = -1.0
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = np.pi * r ** 4 / 2
            P_cr_analytical = np.pi ** 2 * E * I / (2.0 * L) ** 2
            n_nodes = n_elements + 1
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_elements: [0, 0, P_ref, 0, 0, 0]}
            (lambda_num, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_numerical = lambda_num * abs(P_ref)
            assert P_cr_numerical == pytest.approx(P_cr_analytical, rel=0.001)

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
    L = 20.0
    (b, h) = (0.2, 0.4)
    A = b * h
    I_y = h * b ** 3 / 12
    I_z = b * h ** 3 / 12
    J = b * h ** 3 * (1 / 3 - 0.21 * (h / b) * (1 - h ** 4 / (12 * b ** 4)))
    n_elements = 8
    n_nodes = n_elements + 1
    P_ref = -1.0
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_nodes)
    local_z_base = np.array([0.0, 1.0, 0.0])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_base})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_vec_base = np.array([0, 0, P_ref, 0, 0, 0])
    nodal_loads_base = {n_elements: load_vec_base}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    theta = np.deg2rad(45)
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot})
    load_force_rot = R @ load_vec_base[:3]
    load_moment_rot = R @ load_vec_base[3:]
    nodal_loads_rot = {n_elements: np.hstack([load_force_rot, load_moment_rot])}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert lambda_rot == pytest.approx(lambda_base)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    norm_rot = mode_rot / np.linalg.norm(mode_rot)
    norm_base_t = mode_base_transformed / np.linalg.norm(mode_base_transformed)
    assert abs(np.dot(norm_rot, norm_base_t)) == pytest.approx(1.0)

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
    r = 0.1
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = 2 * I
    P_ref = -1.0
    P_cr_analytical = np.pi ** 2 * E * I / (2.0 * L) ** 2
    n_elements_list = [2, 4, 8, 16, 32]
    errors = []
    for n_elements in n_elements_list:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [0, 0, P_ref, 0, 0, 0]}
        (lambda_num, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_numerical = lambda_num * abs(P_ref)
        error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
        errors.append(error)
    assert np.all(np.diff(errors) <= 1e-12)
    assert errors[-1] < 1e-07