def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 210000000000.0
    nu = 0.3
    P_ref = 1.0
    num_elements = 10
    for r in [0.5, 0.75, 1.0]:
        for L in [10, 20, 40]:
            node_coords = np.array([[0, 0, i * L / num_elements] for i in range(num_elements + 1)])
            elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'Iy': np.pi * r ** 4 / 4, 'Iz': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2, 'I_rho': np.pi * r ** 4 / 2, 'local_z': None} for i in range(num_elements)]
            boundary_conditions = {0: [True, True, True, True, True, True]}
            nodal_loads = {num_elements: [0, 0, -P_ref, 0, 0, 0]}
            (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr = lambda_cr * P_ref
            I = np.pi * r ** 4 / 4
            P_euler = np.pi ** 2 * E * I / L ** 2
            assert abs(P_cr - P_euler) / P_euler < 1e-05

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
    L = 10.0
    b = 0.3
    h = 0.5
    num_elements = 10
    P_ref = 1.0
    node_coords = np.array([[0, 0, i * L / num_elements] for i in range(num_elements + 1)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': b * h, 'Iy': b * h ** 3 / 12, 'Iz': h * b ** 3 / 12, 'J': b * h ** 3 / 3, 'I_rho': b * h ** 3 / 3, 'local_z': None} for i in range(num_elements)]
    boundary_conditions = {0: [True, True, True, True, True, True]}
    nodal_loads = {num_elements: [0, 0, -P_ref, 0, 0, 0]}
    (lambda_cr_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    node_coords_rot = node_coords @ R.T
    elements_rot = [{**ele, 'local_z': R @ np.array([0, 0, 1])} for ele in elements]
    nodal_loads_rot = {k: R @ np.array(v[:3]) for (k, v) in nodal_loads.items()}
    (lambda_cr_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isclose(lambda_cr_base, lambda_cr_rot, atol=1e-05)
    T = np.kron(np.eye(num_elements + 1), np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    assert np.allclose(mode_rot, T @ mode_base, atol=1e-05)

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
    r = 0.5
    P_ref = 1.0
    I = np.pi * r ** 4 / 4
    P_euler = np.pi ** 2 * E * I / L ** 2
    for num_elements in [5, 10, 20, 40]:
        node_coords = np.array([[0, 0, i * L / num_elements] for i in range(num_elements + 1)])
        elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': np.pi * r ** 2, 'Iy': np.pi * r ** 4 / 4, 'Iz': np.pi * r ** 4 / 4, 'J': np.pi * r ** 4 / 2, 'I_rho': np.pi * r ** 4 / 2, 'local_z': None} for i in range(num_elements)]
        boundary_conditions = {0: [True, True, True, True, True, True]}
        nodal_loads = {num_elements: [0, 0, -P_ref, 0, 0, 0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr = lambda_cr * P_ref
        relative_error = abs(P_cr - P_euler) / P_euler
        assert relative_error < 1 / num_elements