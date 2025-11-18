def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_elems = 10
    n_nodes = n_elems + 1
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 1.0, 1.0])
    axis_vec = end - start
    L = np.linalg.norm(axis_vec)
    axis_dir = axis_vec / L
    node_coords = np.array([start + i / n_elems * axis_vec for i in range(n_nodes)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-06
    Iy = I
    Iz = I
    J = 2.0 * I
    elements = []
    for i in range(n_elems):
        local_z = np.array([0.0, 0.0, 1.0], dtype=float)
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_mag = 1000.0
    F_dir = np.array([1.0, -1.0, 0.0], dtype=float)
    F_dir = F_dir / np.linalg.norm(F_dir)
    assert abs(np.dot(F_dir, axis_dir)) < 1e-12
    F_vec = F_mag * F_dir
    nodal_loads = {n_nodes - 1: [F_vec[0], F_vec[1], F_vec[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_expected = F_mag * L ** 3 / (3.0 * E * I)
    u_tip = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    delta_projected = float(np.dot(u_tip, F_dir))
    assert np.isclose(delta_projected, delta_expected, rtol=0.01, atol=1e-06)
    axial_disp = float(np.dot(u_tip, axis_dir))
    assert abs(axial_disp) <= max(1e-06, 0.001 * abs(delta_expected))
    assert np.allclose(u[0:6], 0.0, atol=1e-12)
    r_base = r[0:6]
    assert np.allclose(r_base[:3], -F_vec, rtol=1e-08, atol=1e-09)
    r_tip_pos = node_coords[-1] - node_coords[0]
    M_expected = -np.cross(r_tip_pos, F_vec)
    assert np.allclose(r_base[3:6], M_expected, rtol=1e-06, atol=1e-08)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 8e-07
    Iz = 8e-07
    J = 1.6e-06
    conns = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (2, 4)]
    elements = []
    for (i, j) in conns:
        p_i = node_coords[i]
        p_j = node_coords[j]
        axis = p_j - p_i
        axis_norm = np.linalg.norm(axis)
        assert axis_norm > 0
        axis_dir = axis / axis_norm
        if abs(np.dot(axis_dir, np.array([0.0, 0.0, 1.0]))) > 0.9:
            local_z = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            local_z = np.array([0.0, 0.0, 1.0], dtype=float)
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads1 = {2: [500.0, -200.0, 150.0, 0.0, 50.0, 80.0], 4: [0.0, 0.0, -800.0, 30.0, 0.0, -40.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    r_nonfixed = r1[6:]
    assert np.allclose(r_nonfixed, 0.0, atol=1e-09)
    loads2 = {k: (np.array(v, dtype=float) * 2.0).tolist() for (k, v) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-06, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-06, atol=1e-09)
    loads3 = {k: (np.array(v, dtype=float) * -1.0).tolist() for (k, v) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -u1, rtol=1e-06, atol=1e-09)
    assert np.allclose(r3, -r1, rtol=1e-06, atol=1e-09)
    F_applied = np.zeros(3)
    M_applied = np.zeros(3)
    for (n_idx, load) in loads1.items():
        F_applied += np.array(load[:3], dtype=float)
        M_applied += np.array(load[3:6], dtype=float)
    Rf = r1[0:3]
    Rm = r1[3:6]
    assert np.allclose(Rf + F_applied, 0.0, rtol=1e-08, atol=1e-08)
    origin = node_coords[0]
    moment_of_forces = np.zeros(3)
    for (n_idx, load) in loads1.items():
        pos = node_coords[n_idx] - origin
        Fi = np.array(load[:3], dtype=float)
        moment_of_forces += np.cross(pos, Fi)
    M_total_applied = M_applied + moment_of_forces
    assert np.allclose(Rm + M_total_applied, 0.0, rtol=1e-07, atol=1e-08)