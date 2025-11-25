def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements,
    tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    node_coords = np.array([[i, i, i] for i in range(11)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])} for i in range(10)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {10: [0, 1000, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    expected_tip_deflection = -0.001
    assert np.isclose(u[60 + 1], expected_tip_deflection, atol=1e-05), 'Tip deflection does not match analytical solution'

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u, 0), 'Displacements should be zero for zero loads'
    assert np.allclose(r, 0), 'Reactions should be zero for zero loads'
    nodal_loads = {1: [100, 0, 0, 0, 0, 0], 2: [0, 200, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u, 0), 'Displacements should be nonzero for applied loads'
    assert not np.allclose(r, 0), 'Reactions should be nonzero for applied loads'
    nodal_loads = {1: [200, 0, 0, 0, 0, 0], 2: [0, 400, 0, 0, 0, 0]}
    (u_double, r_double) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_double, 2 * u), 'Displacements should double when loads are doubled'
    assert np.allclose(r_double, 2 * r), 'Reactions should double when loads are doubled'
    nodal_loads = {1: [-100, 0, 0, 0, 0, 0], 2: [0, -200, 0, 0, 0, 0]}
    (u_neg, r_neg) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u_neg, -u), 'Displacements should flip sign when loads are negated'
    assert np.allclose(r_neg, -r), 'Reactions should flip sign when loads are negated'
    total_reaction = np.sum(r_neg.reshape(-1, 6), axis=0)
    total_load = np.array([-100, -200, 0, 0, 0, 0])
    assert np.allclose(total_reaction, -total_load), 'Reactions should satisfy global static equilibrium'

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """
    Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0, 0, 1])}]
    boundary_conditions = {}
    nodal_loads = {}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)