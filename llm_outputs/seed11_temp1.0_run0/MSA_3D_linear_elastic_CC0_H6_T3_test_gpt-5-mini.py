def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force
    perpendicular to the beam axis. Verify beam tip deflection with the appropriate analytical reference solution.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.0001
    I_y = 8.333e-10
    I_z = I_y
    J = 1e-09
    n_elems = 10
    n_nodes = n_elems + 1
    L_total = 1.0
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    for i in range(n_nodes):
        node_coords[i] = axis * (L_total * i / n_elems)
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J})
    bc = {0: [1, 1, 1, 1, 1, 1]}
    global_z = np.array([0.0, 0.0, 1.0])
    local_y = np.cross(global_z, axis)
    local_y_norm = np.linalg.norm(local_y)
    if local_y_norm < 1e-12:
        global_y = np.array([0.0, 1.0, 0.0])
        local_y = np.cross(global_y, axis)
        local_y_norm = np.linalg.norm(local_y)
    local_y = local_y / local_y_norm
    P = 100.0
    nodal_loads = {n_nodes - 1: (P * local_y).tolist() + [0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, bc, nodal_loads)
    L = L_total
    I_used = I_z
    delta_expected = P * L ** 3 / (3.0 * E * I_used)
    tip_disp = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    disp_along_load = float(np.dot(tip_disp, local_y))
    assert np.isfinite(disp_along_load)
    assert abs(disp_along_load - delta_expected) / max(abs(delta_expected), 1e-12) < 0.001
    applied_vec = _assemble_nodal_vector(n_nodes, nodal_loads)
    net = r + applied_vec
    assert np.linalg.norm(net) / (np.linalg.norm(applied_vec) + 1e-12) < 1e-09

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    E = 200000000000.0
    nu = 0.3
    A = 0.0005
    I_y = 2e-08
    I_z = 1.5e-08
    J = 5e-09
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.8, 0.9, 0.4], [2.5, 1.6, 0.8], [3.0, 1.0, 1.5]], dtype=float)
    n_nodes = node_coords.shape[0]
    elements = []
    connectivity = [(0, 1), (1, 2), (2, 3), (1, 4)]
    for (ni, nj) in connectivity:
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J})
    bc = {0: [1, 1, 1, 1, 1, 1]}
    zero_loads = {}
    (u0, r0) = fcn(node_coords, elements, bc, zero_loads)
    assert np.allclose(u0, np.zeros_like(u0), atol=1e-12)
    assert np.allclose(r0, np.zeros_like(r0), atol=1e-12)
    loads = {2: [500.0, -200.0, 150.0, 10.0, -5.0, 2.0], 3: [-100.0, 300.0, -50.0, -2.0, 4.0, -1.0], 4: [0.0, 0.0, 250.0, 0.0, 1.0, 0.5]}
    (u1, r1) = fcn(node_coords, elements, bc, loads)
    assert not np.allclose(u1, 0.0)
    assert not np.allclose(r1, 0.0)
    loads_double = {k: (np.array(v) * 2.0).tolist() for (k, v) in loads.items()}
    (u2, r2) = fcn(node_coords, elements, bc, loads_double)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-08, atol=1e-10)
    loads_neg = {k: (np.array(v) * -1.0).tolist() for (k, v) in loads.items()}
    (u3, r3) = fcn(node_coords, elements, bc, loads_neg)
    assert np.allclose(u3, -1.0 * u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r3, -1.0 * r1, rtol=1e-08, atol=1e-10)
    applied_vec = _assemble_nodal_vector(n_nodes, loads)
    net = r1 + applied_vec
    assert np.linalg.norm(net) / (np.linalg.norm(applied_vec) + 1e-12) < 1e-09