def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 3.0
    n_el = 10
    n_nodes = n_el + 1
    axis_dir = np.array([1.0, 1.0, 1.0])
    axis_unit = axis_dir / np.linalg.norm(axis_dir)
    ds = L / n_el
    node_coords = np.array([i * ds * axis_unit for i in range(n_nodes)])
    E = 210000000000.0
    nu = 0.3
    A = 0.001
    I = 8e-06
    Iy = I
    Iz = I
    J = 2.0 * I
    elements = []
    local_z = np.array([0.0, 0.0, 1.0])
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    F_dir = np.array([1.0, -1.0, 0.0])
    F_dir = F_dir / np.linalg.norm(F_dir)
    F_vec = F_mag * F_dir
    nodal_loads = {n_el: [F_vec[0], F_vec[1], F_vec[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_ref = F_mag * L ** 3 / (3.0 * E * I)
    tip = n_el
    u_tip = np.array(u[6 * tip:6 * tip + 3])
    delta_num = np.dot(u_tip, F_dir)
    assert np.isclose(delta_num, delta_ref, rtol=0.0001, atol=1e-09)
    axial_disp = np.dot(u_tip, axis_unit)
    assert np.isclose(axial_disp, 0.0, atol=delta_ref * 1e-05)
    r_base_F = np.array(r[0:3])
    r_base_M = np.array(r[3:6])
    x_tip = node_coords[tip]
    M_expected = -np.cross(x_tip, F_vec)
    assert np.allclose(r_base_F, -F_vec, rtol=1e-10, atol=1e-09)
    assert np.allclose(r_base_M, M_expected, rtol=1e-08, atol=1e-08)
    assert np.allclose(r[6:], 0.0, atol=1e-12)

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
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, L, 0.0], [L, L, L]])
    E = 210000000000.0
    nu = 0.3
    A = 0.002
    Iy = 6e-06
    Iz = 6e-06
    J = 2.0 * Iy

    def pick_local_z(p_i, p_j):
        axis = p_j - p_i
        axis = axis / np.linalg.norm(axis)
        z_cand = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(axis, z_cand)) > 0.99:
            z_cand = np.array([0.0, 1.0, 0.0])
        return z_cand
    elements = []
    conn = [(0, 1), (1, 2), (2, 3)]
    for (i, j) in conn:
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': pick_local_z(node_coords[i], node_coords[j])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u0, 0.0, atol=1e-14)
    assert np.allclose(r0, 0.0, atol=1e-14)
    loads_base = {1: [100.0, -50.0, 80.0, 10.0, 0.0, 5.0], 2: [-60.0, 40.0, -70.0, -8.0, 12.0, 0.0], 3: [0.0, 30.0, 20.0, 0.0, -6.0, 9.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads_base)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1[0:6]) > 0.0
    assert np.allclose(r1[6:], 0.0, atol=1e-12)
    loads_double = {k: (np.array(v) * 2.0).tolist() for (k, v) in loads_base.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads_double)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-09, atol=1e-09)
    loads_neg = {k: (np.array(v) * -1.0).tolist() for (k, v) in loads_base.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u3, -u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r3, -r1, rtol=1e-09, atol=1e-09)
    F_sum = np.zeros(3)
    M_sum_nodal = np.zeros(3)
    for (nid, load) in loads_base.items():
        F_sum += np.array(load[0:3])
        M_sum_nodal += np.array(load[3:6])
    rF = np.array(r1[0:3])
    assert np.allclose(rF + F_sum, np.zeros(3), rtol=1e-10, atol=1e-08)
    M_forces = np.zeros(3)
    for (nid, load) in loads_base.items():
        pos = node_coords[nid]
        F = np.array(load[0:3])
        M_forces += np.cross(pos, F)
    rM = np.array(r1[3:6])
    M_total_ext = M_sum_nodal + M_forces
    assert np.allclose(rM + M_total_ext, np.zeros(3), rtol=1e-09, atol=1e-08)