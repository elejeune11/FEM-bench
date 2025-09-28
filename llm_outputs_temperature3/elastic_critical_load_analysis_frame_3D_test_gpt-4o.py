def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever (fixed-free) circular column aligned with +z. Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}. For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution. Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5."""
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    E = 210000000000.0
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            node_coords = np.array([[0, 0, i * L / 10] for i in range(11)])
            elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': 0.3, 'A': A, 'Iy': I, 'Iz': I, 'J': I, 'I_rho': I, 'local_z': [0, 0, 1]} for i in range(10)]
            boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
            nodal_loads = {10: [0, 0, -1, 0, 0, 0]}
            (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_ref = 1
            P_cr = lambda_cr * P_ref
            P_euler = np.pi ** 2 * E * I / L ** 2
            assert abs(P_cr - P_euler) / P_euler < 1e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance test with a rectangular section (Iy ≠ Iz). The cantilever model is solved in its original orientation and again after applying a rigid-body rotation R to the geometry, element axes, and applied load. The critical load factor λ should be identical in both cases. The buckling mode from the rotated model should equal the base mode transformed by R."""
    E = 210000000000.0
    Iy = 0.0001
    Iz = 0.0002
    node_coords = np.array([[0, 0, i] for i in range(11)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': 0.3, 'A': 0.01, 'Iy': Iy, 'Iz': Iz, 'J': 0.0001, 'I_rho': 0.0001, 'local_z': [0, 0, 1]} for i in range(10)]
    boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
    nodal_loads = {10: [0, 0, -1, 0, 0, 0]}
    (lambda_cr_base, mode_base) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = [{'node_i': ele['node_i'], 'node_j': ele['node_j'], 'E': ele['E'], 'nu': ele['nu'], 'A': ele['A'], 'Iy': ele['Iy'], 'Iz': ele['Iz'], 'J': ele['J'], 'I_rho': ele['I_rho'], 'local_z': R @ np.array(ele['local_z'])} for ele in elements]
    nodal_loads_rot = {k: R @ np.array(v[:3]) for (k, v) in nodal_loads.items()}
    (lambda_cr_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert np.isclose(lambda_cr_base, lambda_cr_rot, atol=1e-05)
    T = np.kron(np.eye(len(node_coords)), np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    mode_base_transformed = T @ mode_base
    assert np.allclose(mode_rot, mode_base_transformed, atol=1e-05)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence for Euler buckling of a fixed–free circular cantilever. The test refines the beam discretization and checks that the numerical critical load approaches the analytical Euler value with decreasing relative error, and that the finest mesh achieves very high accuracy."""
    E = 210000000000.0
    r = 0.5
    L = 10
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    P_euler = np.pi ** 2 * E * I / L ** 2
    for num_elements in [5, 10, 20, 40]:
        node_coords = np.array([[0, 0, i * L / num_elements] for i in range(num_elements + 1)])
        elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': 0.3, 'A': A, 'Iy': I, 'Iz': I, 'J': I, 'I_rho': I, 'local_z': [0, 0, 1]} for i in range(num_elements)]
        boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
        nodal_loads = {num_elements: [0, 0, -1, 0, 0, 0]}
        (lambda_cr, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr = lambda_cr
        relative_error = abs(P_cr - P_euler) / P_euler
        assert relative_error < 1 / num_elements
    assert relative_error < 1e-05