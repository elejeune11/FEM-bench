def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5."""
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    E = 1000000.0
    nu = 0.3
    P_ref = -1.0
    for r in radii:
        for L in lengths:
            n_nodes = 11
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_nodes - 1):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0, 0, P_ref, 0, 0, 0]}
            (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_numerical = lambda_cr * P_ref
            P_cr_analytical = -(np.pi ** 2 * E * np.pi * r ** 4 / 4) / (4 * L ** 2)
            assert np.isclose(P_cr_numerical, P_cr_analytical, atol=0, rtol=0.0001)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R."""
    L = 10.0
    b = 1.0
    h = 2.0
    E = 1000000.0
    nu = 0.3
    P_ref = -1.0
    n_nodes = 11
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_nodes)
    elements = []
    for i in range(n_nodes - 1):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': b * h, 'I_y': b * h ** 3 / 12, 'I_z': h * b ** 3 / 12, 'J': (b * h ** 3 + h * b ** 3) / 12})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0, 0, P_ref, 0, 0, 0]}
    (lambda_cr_base, mode_base) = fcn(node_coords_base, elements, boundary_conditions, nodal_loads)
    theta = np.pi / 3
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    T = block_diag(*[R] * 6 * n_nodes)
    node_coords_rot = np.dot(node_coords_base, R)
    for elem in elements:
        elem['local_z'] = np.dot(R, [0, 0, 1])
    nodal_loads_rot = {n_nodes - 1: np.dot(R, [0, 0, P_ref]) + [0, 0, 0, 0, 0, 0]}
    (lambda_cr_rot, mode_rot) = fcn(node_coords_rot, elements, boundary_conditions, nodal_loads_rot)
    assert np.isclose(lambda_cr_rot, lambda_cr_base, atol=1e-08, rtol=1e-05)
    assert np.allclose(mode_rot, T @ mode_base, atol=1e-08, rtol=0.001)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy."""
    L = 10.0
    r = 0.5
    E = 1000000.0
    nu = 0.3
    P_ref = -1.0
    analytical_P_cr = -(np.pi ** 2 * E * np.pi * r ** 4 / 4) / (4 * L ** 2)
    n_elements_list = [5, 10, 20, 40]
    lambda_cr_list = []
    for n_elements in n_elements_list:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_nodes - 1):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0, 0, P_ref, 0, 0, 0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        lambda_cr_list.append(lambda_cr)
    P_cr_numerical_list = np.array(lambda_cr_list) * P_ref
    rel_errors = np.abs((P_cr_numerical_list - analytical_P_cr) / analytical_P_cr)
    assert np.all(np.diff(rel_errors) <= 0)
    assert rel_errors[-1] < 0.0001