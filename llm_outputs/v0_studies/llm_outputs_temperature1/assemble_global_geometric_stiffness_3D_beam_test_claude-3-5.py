def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric, 
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    import numpy as np
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    ele_props = {'A': 0.01, 'I_rho': 0.0001, 'local_z': [0, 0, 1]}
    elements = [dict(node_i=0, node_j=1, **ele_props), dict(node_i=1, node_j=2, **ele_props)]
    n_dof = len(nodes) * 6
    u_zero = np.zeros(n_dof)
    K_zero = fcn(nodes, elements, u_zero)
    assert np.allclose(K_zero, 0)
    u_test = np.random.rand(n_dof)
    K = fcn(nodes, elements, u_test)
    assert np.allclose(K, K.T)
    scale = 2.5
    K_scaled = fcn(nodes, elements, scale * u_test)
    assert np.allclose(K_scaled, scale * K)
    u1 = np.random.rand(n_dof)
    u2 = np.random.rand(n_dof)
    K1 = fcn(nodes, elements, u1)
    K2 = fcn(nodes, elements, u2)
    K_sum = fcn(nodes, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2)
    elements_reversed = elements[::-1]
    K_rev = fcn(nodes, elements_reversed, u_test)
    assert np.allclose(K, K_rev)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    ele_props = {'A': 0.01, 'I_rho': 0.0001, 'local_z': [0, 0, 1]}
    elements = [dict(node_i=0, node_j=1, **ele_props), dict(node_i=1, node_j=2, **ele_props)]
    n_nodes = len(nodes)
    n_dof = n_nodes * 6
    u_orig = np.random.rand(n_dof)
    K_orig = fcn(nodes, elements, u_orig)
    R = Rotation.random().as_matrix()
    nodes_rot = nodes @ R.T
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rot = T @ u_orig
    K_rot = fcn(nodes_rot, elements, u_rot)
    K_transformed = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_transformed, rtol=1e-10)