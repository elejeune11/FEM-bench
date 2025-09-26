def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    node_coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])} for i in range(10)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [0, 1000, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    expected_tip_deflection = -0.001
    assert np.isclose(u[-5], expected_tip_deflection, atol=1e-05)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0)
    assert np.allclose(r, 0)
    nodal_loads = {1: [1000, 0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 1000, 0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    nodal_loads = {1: [2000, 0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 2000, 0]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    nodal_loads = {1: [-1000, 0, 0, 0, 0, 0], 2: [0, 0, 0, 0, -1000, 0]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u3, -u1)
    assert np.allclose(r3, -r1)
    total_reaction = np.sum(r1.reshape(-1, 6), axis=0)
    total_load = np.array([1000, 0, 0, 0, 1000, 0])
    assert np.allclose(total_reaction, -total_load)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """
    Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {}
    nodal_loads = {1: [1000, 0, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)