def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force
    perpendicular to the beam axis. Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_elems = 10
    node_coords = np.array([i / n_elems * np.array([1.0, 1.0, 1.0]) for i in range(n_elems + 1)], dtype=float)
    N = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 1.0
    I = 2.5e-05
    J = 2.0 * I
    elements = []
    for i in range(n_elems):
        elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I, I_z=I, J=J))
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    load_dir = np.array([1.0, -1.0, 0.0])
    load_dir /= np.linalg.norm(load_dir)
    F_vec = P * load_dir
    nodal_loads = {N - 1: [F_vec[0], F_vec[1], F_vec[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u_tip = u[6 * (N - 1):6 * (N - 1) + 3]
    L_total = np.linalg.norm(node_coords[-1] - node_coords[0])
    c = L_total ** 3 / (3.0 * E * I)
    u_tip_ref = c * F_vec
    assert np.allclose(u_tip, u_tip_ref, rtol=0.002, atol=1e-09)
    axis_unit = (node_coords[-1] - node_coords[0]) / L_total
    axial_comp = np.dot(u_tip, axis_unit)
    assert abs(axial_comp) <= 1e-08 + 1e-06 * np.linalg.norm(u_tip_ref)
    r_force = np.array([np.sum(u * 0 + r[0::6]), np.sum(u * 0 + r[1::6]), np.sum(u * 0 + r[2::6])])
    assert np.allclose(r_force, -F_vec, rtol=1e-10, atol=1e-10)
    r_base = node_coords[0]
    r_tip = node_coords[-1]
    expected_moment_base = -np.cross(r_tip - r_base, F_vec)
    m_base = r[3:6]
    assert np.allclose(m_base, expected_moment_base, rtol=1e-10, atol=1e-10)

def test_complex_geometry_and_basic_loading(fcn):
    """
    Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [2.2, 0.8, 0.6], [1.8, 1.5, 1.8], [0.5, 1.7, 2.4]], dtype=float)
    N = node_coords.shape[0]
    E = 200000000000.0
    nu = 0.29
    A = 0.02
    I = 8e-06
    J = 2.0 * I
    elements = []
    for i in range(N - 1):
        elements.append(dict(node_i=i, node_j=i + 1, E=E, nu=nu, A=A, I_y=I, I_z=I, J=J))
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_0 = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, nodal_loads_0)
    assert np.allclose(u0, 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(r0, 0.0, atol=1e-12, rtol=0.0)
    nodal_loads_1 = {2: [100.0, -80.0, 60.0, 0.0, 5.0, -7.0], 3: [-50.0, 70.0, 0.0, 3.0, 0.0, 5.0], 4: [0.0, 0.0, -120.0, 0.0, 0.0, 0.0]}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert np.isfinite(u1).all() and np.isfinite(r1).all()
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    assert np.allclose(u1[0:6], 0.0, atol=1e-12, rtol=0.0)
    nodal_loads_2 = {k: [2.0 * v for v in vals] for k, vals in nodal_loads_1.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-10, atol=1e-10)
    nodal_loads_neg = {k: [-v for v in vals] for k, vals in nodal_loads_1.items()}
    u_neg, r_neg = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r_neg, -r1, rtol=1e-10, atol=1e-10)
    sum_react_force = np.array([np.sum(r1[0::6]), np.sum(r1[1::6]), np.sum(r1[2::6])])
    sum_react_moment = np.array([np.sum(r1[3::6]), np.sum(r1[4::6]), np.sum(r1[5::6])])
    sum_applied_force = np.zeros(3)
    sum_applied_moment = np.zeros(3)
    moment_from_forces = np.zeros(3)
    origin = np.array([0.0, 0.0, 0.0])
    for n_idx, load in nodal_loads_1.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:], dtype=float)
        pos = node_coords[n_idx]
        sum_applied_force += F
        sum_applied_moment += M
        moment_from_forces += np.cross(pos - origin, F)
    assert np.allclose(sum_react_force + sum_applied_force, np.zeros(3), rtol=1e-10, atol=1e-10)
    assert np.allclose(sum_react_moment + sum_applied_moment + moment_from_forces, np.zeros(3), rtol=1e-10, atol=1e-10)