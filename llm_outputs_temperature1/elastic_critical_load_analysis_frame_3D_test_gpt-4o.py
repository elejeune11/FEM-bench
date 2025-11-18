def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 210000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    num_elements = 10
    for r in radii:
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4
        for L in lengths:
            node_coords = np.array([[0, 0, i * L / num_elements] for i in range(num_elements + 1)])
            elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': I, 'I_rho': I, 'local_z': [0, 0, 1]} for i in range(num_elements)]
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {num_elements: [0, 0, -1, 0, 0, 0]}
            (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_ref = np.array([0, 0, -1, 0, 0, 0])
            P_cr = lambda_cr * P_ref[2]
            euler_critical_load = np.pi ** 2 * E * I / L ** 2
            assert np.isclose(P_cr, euler_critical_load, rtol=1e-05), f'Failed for r={r}, L={L}'

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
    L = 10
    A = 0.1
    I_y = 0.01
    I_z = 0.02
    num_elements = 10
    node_coords = np.array([[0, 0, i * L / num_elements] for i in range(num_elements + 1)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': I_y + I_z, 'I_rho': I_y + I_z, 'local_z': [0, 0, 1]} for i in range(num_elements)]
    boundary_conditions = {0: [True, True, True, True, True, True]}
    nodal_loads = {num_elements: [0, 0, -1, 0, 0, 0]}
    (lambda_cr_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    T = np.kron(np.eye(num_elements + 1), np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = [{'node_i': e['node_i'], 'node_j': e['node_j'], 'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'I_rho': e['I_rho'], 'local_z': R @ np.array(e['local_z'])} for e in elements]
    nodal_loads_rot = {k: R @ np.array(v[:3]) for (k, v) in nodal_loads.items()}
    (lambda_cr_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isclose(lambda_cr_base, lambda_cr_rot, rtol=1e-05), 'Critical load factors do not match after rotation'
    assert np.allclose(mode_rot, T @ mode_base, atol=1e-05), 'Buckling modes do not match after rotation'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.5
    L = 10
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    num_elements_list = [5, 10, 20, 40]
    euler_critical_load = np.pi ** 2 * E * I / L ** 2
    previous_error = float('inf')
    for num_elements in num_elements_list:
        node_coords = np.array([[0, 0, i * L / num_elements] for i in range(num_elements + 1)])
        elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': I, 'I_rho': I, 'local_z': [0, 0, 1]} for i in range(num_elements)]
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {num_elements: [0, 0, -1, 0, 0, 0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_ref = np.array([0, 0, -1, 0, 0, 0])
        P_cr = lambda_cr * P_ref[2]
        relative_error = abs((P_cr - euler_critical_load) / euler_critical_load)
        assert relative_error < previous_error, f'Mesh refinement did not improve accuracy for {num_elements} elements'
        previous_error = relative_error
    assert relative_error < 1e-05, 'Finest mesh did not achieve high accuracy'