def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10^-5.
    """
    for r in [0.5, 0.75, 1.0]:
        for L in [10, 20, 40]:
            node_coords = np.array([[0, 0, 0], [0, 0, L]])
            elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': 2 * np.pi * r ** 4 / 3}]
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [0, 0, 0, 0, 0, 0]}
            nodal_loads = {1: [0, 0, -1, 0, 0, 0]}
            (lambda_crit, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            euler_lambda = 2.046 * (200000000000.0 * np.pi * r ** 4 / 4) / L ** 2
            assert np.isclose(lambda_crit * 1, euler_lambda, rtol=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    node_coords = np.array([[0, 0, 0], [0, 0, 10]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.1 * 0.2, 'I_y': 0.1 * 0.2 ** 3 / 12, 'I_z': 0.2 * 0.1 ** 3 / 12, 'J': 0.1 * 0.2 ** 3 / 3}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [0, 0, 0, 0, 0, 0]}
    nodal_loads = {1: [0, 0, -1, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    rotation = R.from_euler('xyz', [np.pi / 4, 0, 0])
    node_coords_rot = rotation.apply(node_coords)
    elements_rot = [{**elem, 'local_z': rotation.apply(elem.get('local_z', [0, 0, 1]))} for elem in elements]
    nodal_loads_rot = {node: rotation.apply(load) for (node, load) in nodal_loads.items()}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isclose(lambda_base, lambda_rot)
    T = np.eye(6)
    for i in range(3):
        T[np.ix_(i, i)] = rotation.as_matrix()
    T = np.block([[T, np.zeros((3, 3))], [np.zeros((3, 3)), T]])
    mode_base_scaled = mode_base / mode_base[0]
    mode_rot_scaled = mode_rot / mode_rot[0]
    assert np.allclose(mode_rot_scaled, T @ mode_base_scaled)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 10
    r = 0.5
    euler_lambda = 2.046 * (200000000000.0 * np.pi * r ** 4 / 4) / L ** 2
    for n_elements in [1, 2, 4, 8, 16]:
        node_coords = np.zeros((n_elements + 1, 3))
        node_coords[:, 2] = np.linspace(0, L, n_elements + 1)
        elements = [{'node_i': i, 'node_j': i + 1, 'E': 200000000000.0, 'nu': 0.3, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': 2 * np.pi * r ** 4 / 3} for i in range(n_elements)]
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [0, 0, -1, 0, 0, 0]}
        (lambda_crit, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        assert np.isclose(lambda_crit * 1, euler_lambda, rtol=0.001 * (1 / n_elements))
    assert np.isclose(lambda_crit * 1, euler_lambda, rtol=1e-06)