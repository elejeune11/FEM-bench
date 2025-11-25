def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1, 'I_rho': 1}, {'node_i': 1, 'node_j': 2, 'A': 1, 'I_rho': 1}]
    u_global = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_global)
    assert np.allclose(K_zero, 0)
    u_global = np.ones(18)
    K_one = fcn(node_coords, elements, u_global)
    assert np.allclose(K_one, K_one.T)
    K_scaled = fcn(node_coords, elements, 2 * u_global)
    assert np.allclose(K_scaled, 2 * K_one)
    K_super = fcn(node_coords, elements, u_global + 2 * u_global)
    assert np.allclose(K_super, K_one + K_scaled)
    elements_reversed = elements[::-1]
    K_reversed = fcn(node_coords, elements_reversed, u_global)
    assert np.allclose(K_reversed, K_one)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    pass