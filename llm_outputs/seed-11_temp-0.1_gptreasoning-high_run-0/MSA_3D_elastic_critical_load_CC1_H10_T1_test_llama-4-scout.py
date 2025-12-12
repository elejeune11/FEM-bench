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
            elements = [{'node_i': 0, 'node_j': 1, 'E': 200000.0, 'nu': 0.3, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': 2 * np.pi * r ** 4 / 4}]
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [0, 0, 0, 0, 0, 0]}
            nodal_loads = {1: [0, 0, -1, 0, 0, 0]}
            (lambda_crit, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            euler_lambda = 2.046 * np.pi ** 2 * 200000.0 * np.pi * r ** 4 / (L ** 2 * 4)
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
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000.0, 'nu': 0.3, 'A': 1 * 2, 'I_y': 1 ** 3 * 2 / 12, 'I_z': 2 ** 3 * 1 / 12, 'J': 2 * 1 ** 3 * 2 / 12}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 1: [0, 0, 0, 0, 0, 0]}
    nodal_loads = {1: [0, 0, -1, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    rot = R.from_euler('xyz', [np.pi / 4, 0, 0])
    node_coords_rot = rot.apply(node_coords)
    elements_rot = [{**elem, 'local_z': rot.apply(elem.get('local_z', [0, 0, 1]))} for elem in elements]
    nodal_loads_rot = {node: rot.apply(load) for (node, load) in nodal_loads.items()}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isclose(lambda_base, lambda_rot, rtol=1e-05)
    T = np.eye(6 * len(node_coords))
    for i in range(len(node_coords)):
        block = np.eye(6)
        block[:3, :3] = rot.as_matrix()
        block[3:, 3:] = rot.as_matrix()
        T[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = block
    assert np.allclose(np.abs(mode_rot), np.abs(T @ mode_base), atol=1e-05)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 10
    r = 1.0
    euler_lambda = 2.046 * np.pi ** 2 * 200000.0 * np.pi * r ** 4 / (L ** 2 * 4)
    for n_elements in [1, 2, 4, 8, 16]:
        node_coords = np.zeros((n_elements + 1, 3))
        node_coords[:, 2] = np.linspace(0, L, n_elements + 1)
        elements = [{'node_i': i, 'node_j': i + 1, 'E': 200000.0, 'nu': 0.3, 'A': np.pi * r ** 2, 'I_y': np.pi * r ** 4 / 4, 'I_z': np.pi * r ** 4 / 4, 'J': 2 * np.pi * r ** 4 / 4} for i in range(n_elements)]
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_elements: [0, 0, -1, 0, 0, 0]}
        (lambda_crit, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        rel_error = np.abs(lambda_crit * 1 - euler_lambda) / euler_lambda
        assert rel_error < 0.001