def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 1000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    n_elements = 10
    for r in radii:
        for L in lengths:
            node_coords = np.linspace([0, 0, 0], [0, 0, L], n_elements + 1)
            elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2} for i in range(n_elements)]
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_elements: [-1, 0, 0, 0, 0, 0]}
            (elastic_critical_load_factor, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            euler_load = np.pi ** 2 * E * np.pi * r ** 4 / 4 / (4 * L ** 2)
            assert np.isclose(elastic_critical_load_factor, euler_load, atol=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 1000000.0
    nu = 0.3
    L = 10
    b = 1
    h = 2
    n_elements = 10
    node_coords = np.linspace([0, 0, 0], [0, 0, L], n_elements + 1)
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': b * h, 'I_y': b * h ** 3 / 12, 'I_z': h * b ** 3 / 12, 'J': (b * h ** 3 + h * b ** 3) / 12} for i in range(n_elements)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_elements: [-1, 0, 0, 0, 0, 0]}
    (elastic_critical_load_factor_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    T = block_diag(*[R] * 2 * (n_elements + 1))
    node_coords_rotated = np.dot(node_coords, R)
    elements_rotated = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': b * h, 'I_y': b * h ** 3 / 12, 'I_z': h * b ** 3 / 12, 'J': (b * h ** 3 + h * b ** 3) / 12, 'local_z': np.dot(R, [0, 0, 1])} for i in range(n_elements)]
    nodal_loads_rotated = {n_elements: np.dot(R, [-1, 0, 0]) + [0, 0, 0]}
    (elastic_critical_load_factor_rotated, mode_rotated) = fcn(node_coords_rotated, elements_rotated, boundary_conditions, {k: v for (k, v) in nodal_loads_rotated.items()})
    assert np.isclose(elastic_critical_load_factor_base, elastic_critical_load_factor_rotated, atol=1e-08)
    assert np.allclose(np.abs(mode_rotated), np.abs(T @ mode_base), atol=1e-08)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 1000000.0
    nu = 0.3
    r = 1
    L = 10
    n_elements_list = [5, 10, 20, 40]
    euler_load = np.pi ** 2 * E * np.pi * r ** 4 / 4 / (4 * L ** 2)
    for n_elements in n_elements_list:
        node_coords = np.linspace([0, 0, 0], [0, 0, L], n_elements + 1)
        elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2} for i in range(n_elements)]
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [-1, 0, 0, 0, 0, 0]}
        (elastic_critical_load_factor, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        relative_error = np.abs((elastic_critical_load_factor - euler_load) / euler_load)
        if n_elements == n_elements_list[-1]:
            assert relative_error < 1e-06
        else:
            assert relative_error < 1 / n_elements ** 2