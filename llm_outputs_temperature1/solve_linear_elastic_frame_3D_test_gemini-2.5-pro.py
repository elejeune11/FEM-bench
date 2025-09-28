def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 10.0
    E = 200000000000.0
    nu = 0.3
    h = 0.1
    A = h ** 2
    I = h ** 4 / 12
    J = 0.1406 * h ** 4
    P = 1000.0
    n_elem = 10
    n_nodes = n_elem + 1
    v_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    node_coords = np.array([i * (L / n_elem) * v_axis for i in range(n_nodes)])
    local_z = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    element_props = {'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}
    elements = [{'node_i': i, 'node_j': i + 1, **element_props} for i in range(n_elem)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    local_y = np.cross(local_z, v_axis)
    force_vec = P * local_y
    nodal_loads = {n_nodes - 1: np.concatenate((force_vec, [0.0, 0.0, 0.0]))}
    analytical_delta = P * L ** 3 / (3 * E * I)
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp_vec = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    tip_disp_mag = np.linalg.norm(tip_disp_vec)
    assert tip_disp_mag == pytest.approx(analytical_delta, rel=0.02)
    assert np.allclose(tip_disp_vec / tip_disp_mag, local_y)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [4.0, 0.0, 5.0], [4.0, 3.0, 5.0]])
    n_nodes = len(node_coords)
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.41e-05}
    elements = [{'node_i': 0, 'node_j': 1, **props, 'local_z': np.array([1.0, 0.0, 0.0])}, {'node_i': 1, 'node_j': 2, **props, 'local_z': np.array([0.0, 1.0, 0.0])}, {'node_i': 2, 'node_j': 3, **props, 'local_z': np.array([0.0, 0.0, 1.0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads={})
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    load_vec = np.array([100.0, 200.0, -50.0, 10.0, -20.0, 30.0])
    nodal_loads_1 = {3: load_vec}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert not np.allclose(u1, 0.0)
    assert not np.allclose(r1, 0.0)
    assert np.allclose(u1[0:6], 0.0)
    assert np.allclose(r1[6:], 0.0)
    nodal_loads_2 = {3: 2.0 * load_vec}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2.0 * u1)
    assert np.allclose(r2, 2.0 * r1)
    nodal_loads_3 = {3: -1.0 * load_vec}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_3)
    assert np.allclose(u3, -1.0 * u1)
    assert np.allclose(r3, -1.0 * r1)
    applied_forces = load_vec[0:3]
    reaction_forces = r1[0:3]
    assert np.allclose(applied_forces + reaction_forces, 0.0, atol=1e-09)
    applied_moments = load_vec[3:6]
    reaction_moments = r1[3:6]
    load_application_point = node_coords[3]
    moment_from_applied_force = np.cross(load_application_point, applied_forces)
    total_moments = applied_moments + reaction_moments + moment_from_applied_force
    assert np.allclose(total_moments, 0.0, atol=1e-09)