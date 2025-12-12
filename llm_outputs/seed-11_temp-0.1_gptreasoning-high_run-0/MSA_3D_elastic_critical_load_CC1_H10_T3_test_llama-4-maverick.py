def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever (fixed-free) circular column aligned with +z. Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}. For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution. Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5."""
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    E = 1000000.0
    nu = 0.3
    for r in radii:
        for L in lengths:
            n_nodes = 11
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            elements = []
            for i in range(n_nodes - 1):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0, 0, -1, 0, 0, 0]}
            (elastic_critical_load_factor, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            euler_load = np.pi ** 2 * E * np.pi * r ** 4 / 4 / (4 * L ** 2)
            assert np.isclose(elastic_critical_load_factor, euler_load, atol=0, rtol=0.0001)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance test with a rectangular section (Iy ≠ Iz). The cantilever model is solved in its original orientation and again after applying a rigid-body rotation R to the geometry, element axes, and applied load. The critical load factor λ should be identical in both cases."""
    L = 10.0
    b = 1.0
    h = 2.0
    E = 1000000.0
    nu = 0.3
    n_nodes = 11
    node_coords = np.zeros((n_nodes, 3))
    node_coords[:, 2] = np.linspace(0, L, n_nodes)
    elements = []
    for i in range(n_nodes - 1):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': b * h, 'I_y': b * h ** 3 / 12, 'I_z': h * b ** 3 / 12, 'J': (b * h ** 3 + h * b ** 3) / 12, 'local_z': [1, 0, 0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0, 0, -1, 0, 0, 0]}
    (elastic_critical_load_factor_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    R = np.array([[0.70710678, -0.70710678, 0], [0.70710678, 0.70710678, 0], [0, 0, 1]])
    node_coords_rotated = np.dot(node_coords, R)
    elements_rotated = []
    for elem in elements:
        elem_rotated = elem.copy()
        elem_rotated['local_z'] = np.dot(R, elem['local_z'])
        elements_rotated.append(elem_rotated)
    nodal_loads_rotated = {k: np.dot(R, v[:3]).tolist() + np.dot(R, v[3:]).tolist() for (k, v) in nodal_loads.items()}
    (elastic_critical_load_factor_rotated, mode_rotated) = fcn(node_coords_rotated, elements_rotated, boundary_conditions, nodal_loads_rotated)
    T = block_diag(*[R] * 6 * n_nodes)
    assert np.isclose(elastic_critical_load_factor_base, elastic_critical_load_factor_rotated, atol=1e-08, rtol=1e-05)
    assert np.allclose(mode_rotated, T @ mode_base, atol=1e-08, rtol=0.001) or np.allclose(mode_rotated, -T @ mode_base, atol=1e-08, rtol=0.001)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence for Euler buckling of a fixed–free circular cantilever. The test refines the beam discretization and checks that the numerical critical load approaches the analytical Euler value with decreasing relative error, and that the finest mesh achieves very high accuracy."""
    L = 10.0
    r = 0.5
    E = 1000000.0
    nu = 0.3
    euler_load = np.pi ** 2 * E * np.pi * r ** 4 / 4 / (4 * L ** 2)
    num_elements_list = [5, 10, 20, 40]
    for num_elements in num_elements_list:
        n_nodes = num_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_nodes - 1):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2})
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0, 0, -1, 0, 0, 0]}
        (elastic_critical_load_factor, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        relative_error = np.abs((elastic_critical_load_factor - euler_load) / euler_load)
        if num_elements == num_elements_list[-1]:
            assert relative_error < 1e-06
        if num_elements > num_elements_list[0]:
            previous_relative_error = np.abs((fcn(np.zeros((num_elements_list[num_elements_list.index(num_elements) - 1] + 1, 3)), [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2} for i in range(num_elements_list[num_elements_list.index(num_elements) - 1])], {0: [1, 1, 1, 1, 1, 1]}, {num_elements_list[num_elements_list.index(num_elements) - 1]: [0, 0, -1, 0, 0, 0]})[0] - euler_load) / euler_load)
            assert relative_error < previous_relative_error