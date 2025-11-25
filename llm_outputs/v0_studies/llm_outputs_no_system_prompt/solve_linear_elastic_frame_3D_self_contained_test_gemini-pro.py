def test_simple_beam_discretized_axis_111(fcn):
    """Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements,
    tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 10.0
    E = 1.0
    nu = 0.3
    A = 1.0
    Iy = 1.0
    Iz = 1.0
    J = 1.0
    P = 1.0
    n_elements = 10
    node_coords = np.array([[i * L / n_elements, i * L / n_elements, i * L / n_elements] for i in range(n_elements + 1)])
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': None} for i in range(n_elements)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_elements: [0, P, 0, 0, 0, 0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_uy_tip = P * L ** 3 / (3 * E * Iy)
    np.testing.assert_allclose(u[n_elements * 6 + 1], analytical_uy_tip, rtol=0.001)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}, {'node_i': 2, 'node_j': 3, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}, {'node_i': 3, 'node_j': 0, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}, {'node_i': 0, 'node_j': 4, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}]
    boundary_conditions = {0: [1] * 6, 1: [1, 0, 1, 0, 1, 0]}
    nodal_loads = {}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    np.testing.assert_allclose(u, 0)
    np.testing.assert_allclose(r, 0)
    nodal_loads = {2: [1, 2, 3, 4, 5, 6], 4: [7, 8, 9, 10, 11, 12]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u1, 0)
    assert not np.allclose(r1, 0)
    nodal_loads = {2: [2, 4, 6, 8, 10, 12], 4: [14, 16, 18, 20, 22, 24]}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    np.testing.assert_allclose(u2, 2 * u1)
    np.testing.assert_allclose(r2, 2 * r1)
    nodal_loads = {2: [-1, -2, -3, -4, -5, -6], 4: [-7, -8, -9, -10, -11, -12]}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    np.testing.assert_allclose(u3, -u1)
    np.testing.assert_allclose(r3, -r1)
    np.testing.assert_allclose(np.sum(r1.reshape(-1, 6), axis=0) + np.sum(np.array(list(nodal_loads.values())), axis=0), 0, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test ValueError on ill-conditioned K_ff due to under-constrained structure."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1, 'local_z': None}]
    boundary_conditions = {}
    nodal_loads = {1: [0, 1, 0, 0, 0, 0]}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)