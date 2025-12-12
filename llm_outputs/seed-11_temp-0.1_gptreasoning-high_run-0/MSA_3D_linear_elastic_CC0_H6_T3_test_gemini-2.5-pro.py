def test_simple_beam_discretized_axis_111(fcn: Callable):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 10.0
    n_elem = 10
    n_nodes = n_elem + 1
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    P_val = 1000.0
    p_start = np.array([0.0, 0.0, 0.0])
    v_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    p_end = p_start + L * v_axis
    node_coords = np.linspace(p_start, p_end, n_nodes)
    local_z_dir = np.array([1.0, -1.0, 0.0])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_dir})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    local_x_dir = v_axis
    local_z_norm = local_z_dir / np.linalg.norm(local_z_dir)
    local_y_dir = np.cross(local_z_norm, local_x_dir)
    force_vec = P_val * local_y_dir
    nodal_loads = {n_nodes - 1: np.hstack([force_vec, [0, 0, 0]])}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    analytical_deflection = P_val * L ** 3 / (3 * E * I_z)
    tip_disp_vec = u[(n_nodes - 1) * 6:(n_nodes - 1) * 6 + 3]
    computed_deflection = np.linalg.norm(tip_disp_vec)
    assert computed_deflection == pytest.approx(analytical_deflection, rel=0.001)

def test_complex_geometry_and_basic_loading(fcn: Callable):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    n_nodes = len(node_coords)
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = [{'node_i': 0, 'node_j': 1, 'local_z': [1, 0, 0], **props}, {'node_i': 1, 'node_j': 2, 'local_z': [0, 0, 1], **props}, {'node_i': 3, 'node_j': 2, 'local_z': [1, 0, 0], **props}, {'node_i': 2, 'node_j': 4, 'local_z': [1, 0, 0], **props}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 3: [1, 1, 1, 1, 1, 1]}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads={})
    assert np.allclose(u0, 0.0)
    assert np.allclose(r0, 0.0)
    loads_1 = {4: [1000.0, 2000.0, -500.0, 300.0, -400.0, 600.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads_1)
    assert not np.allclose(u1, 0.0)
    assert not np.allclose(r1, 0.0)
    loads_2 = {k: 2 * np.array(v) for (k, v) in loads_1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads_2)
    assert np.allclose(u2, 2 * u1)
    assert np.allclose(r2, 2 * r1)
    loads_3 = {k: -1 * np.array(v) for (k, v) in loads_1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads_3)
    assert np.allclose(u3, -1 * u1)
    assert np.allclose(r3, -1 * r1)
    sum_forces = np.zeros(3)
    sum_moments_origin = np.zeros(3)
    for (node_idx, load_vec) in loads_1.items():
        forces = load_vec[0:3]
        moments = load_vec[3:6]
        sum_forces += forces
        sum_moments_origin += moments + np.cross(node_coords[node_idx], forces)
    reactions_reshaped = r1.reshape((n_nodes, 6))
    for node_idx in range(n_nodes):
        reaction_vec = reactions_reshaped[node_idx]
        forces = reaction_vec[0:3]
        moments = reaction_vec[3:6]
        sum_forces += forces
        sum_moments_origin += moments + np.cross(node_coords[node_idx], forces)
    assert np.allclose(sum_forces, 0.0, atol=1e-06)
    assert np.allclose(sum_moments_origin, 0.0, atol=1e-06)