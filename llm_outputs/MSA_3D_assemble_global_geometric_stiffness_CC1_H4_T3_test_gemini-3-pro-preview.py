def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001, 'local_z': np.array([0.0, 1.0, 0.0])}
    elements = [{**props, 'connectivity': [0, 1]}, {**props, 'connectivity': [1, 2]}]
    n_nodes = 3
    dof = 6 * n_nodes
    u_zero = np.zeros(dof)
    Kg_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(Kg_zero, 0.0), 'Geometric stiffness should be zero when displacements are zero.'
    rng = np.random.default_rng(42)
    u_a = rng.standard_normal(dof) * 0.001
    u_b = rng.standard_normal(dof) * 0.001
    Kg_a = fcn(node_coords, elements, u_a)
    assert np.allclose(Kg_a, Kg_a.T), 'Geometric stiffness matrix must be symmetric.'
    scale = 2.5
    Kg_scaled = fcn(node_coords, elements, scale * u_a)
    assert np.allclose(Kg_scaled, scale * Kg_a), 'K_g should scale linearly with displacement.'
    Kg_b = fcn(node_coords, elements, u_b)
    Kg_sum = fcn(node_coords, elements, u_a + u_b)
    assert np.allclose(Kg_sum, Kg_a + Kg_b), 'Superposition should hold for K_g assembly w.r.t displacements.'
    elements_reversed = elements[::-1]
    Kg_reversed = fcn(node_coords, elements_reversed, u_a)
    assert np.allclose(Kg_a, Kg_reversed), 'Matrix assembly should be independent of element order.'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    local_z = np.array([0.0, 1.0, 0.0])
    elem_props = {'connectivity': [0, 1], 'E': 10000000.0, 'nu': 0.25, 'A': 0.05, 'I_y': 0.001, 'I_z': 0.002, 'J': 0.003, 'local_z': local_z}
    elements = [elem_props]
    rng = np.random.default_rng(99)
    u_orig = rng.standard_normal(12) * 0.01
    Kg_orig = fcn(node_coords, elements, u_orig)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ local_z
    elem_props_rot = elem_props.copy()
    elem_props_rot['local_z'] = local_z_rot
    elements_rot = [elem_props_rot]
    T_node = np.zeros((6, 6))
    T_node[0:3, 0:3] = R
    T_node[3:6, 3:6] = R
    T_global = np.block([[T_node, np.zeros((6, 6))], [np.zeros((6, 6)), T_node]])
    u_rot = T_global @ u_orig
    Kg_rot_computed = fcn(node_coords_rot, elements_rot, u_rot)
    Kg_rot_predicted = T_global @ Kg_orig @ T_global.T
    assert np.allclose(Kg_rot_computed, Kg_rot_predicted, atol=1e-08, rtol=1e-05), 'Geometric stiffness matrix failed frame objectivity test under rotation.'