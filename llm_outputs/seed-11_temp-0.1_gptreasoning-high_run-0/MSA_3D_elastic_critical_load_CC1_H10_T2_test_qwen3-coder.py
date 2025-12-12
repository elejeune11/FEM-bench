def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4
        for L in lengths:
            n_nodes = 11
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_nodes - 1):
                elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I}
                elements.append(elem)
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0, 0, -1.0, 0, 0, 0]}
            (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_ref = np.array([0, 0, -1.0, 0, 0, 0])
            P_cr_numerical = lambda_cr * P_ref[2]
            P_cr_euler = -(np.pi ** 2 * E * I) / (4 * L ** 2)
            assert abs(P_cr_numerical - P_cr_euler) / abs(P_cr_euler) < 0.001

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
    (b, h) = (0.1, 0.2)
    L = 5.0
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = Iy + Iz
    n_nodes = 11
    node_coords = np.zeros((n_nodes, 3))
    node_coords[:, 2] = np.linspace(0, L, n_nodes)
    elements = []
    for i in range(n_nodes - 1):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J}
        elements.append(elem)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0, 0, -1.0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    angle = np.pi / 4
    (c, s) = (np.cos(angle), np.sin(angle))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for elem in elements:
        new_elem = elem.copy()
        new_elem['local_z'] = R @ np.array([0, 0, 1])
        elements_rot.append(new_elem)
    nodal_loads_rot = {n_nodes - 1: R @ np.array([0, 0, -1.0, 0, 0, 0])}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert abs(lambda_base - lambda_rot) / abs(lambda_base) < 1e-10
    T = np.kron(np.eye(n_nodes), np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    transformed_mode = T @ mode_base
    norm_base = np.linalg.norm(transformed_mode)
    norm_rot = np.linalg.norm(mode_rot)
    if norm_base > 1e-12 and norm_rot > 1e-12:
        transformed_mode /= norm_base
        mode_rot /= norm_rot
        cos_angle = np.dot(transformed_mode, mode_rot)
        assert abs(abs(cos_angle) - 1.0) < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    r = 0.5
    L = 20.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    P_cr_euler = -(np.pi ** 2 * E * I) / (4 * L ** 2)
    mesh_sizes = [4, 8, 16, 32]
    errors = []
    for n_elem in mesh_sizes:
        n_nodes = n_elem + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_elem):
            elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I}
            elements.append(elem)
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0, 0, -1.0, 0, 0, 0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_numerical = lambda_cr * -1.0
        error = abs(P_cr_numerical - P_cr_euler) / abs(P_cr_euler)
        errors.append(error)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]
    assert errors[-1] < 1e-05