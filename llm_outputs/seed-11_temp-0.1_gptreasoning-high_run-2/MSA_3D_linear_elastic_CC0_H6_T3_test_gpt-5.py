def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 8.333e-06
    J = 2 * I
    N_elems = 10
    L_total = 3.0
    e_axis = np.array([1.0, 1.0, 1.0])
    e_axis /= np.linalg.norm(e_axis)
    N_nodes = N_elems + 1
    dL = L_total / N_elems
    node_coords = np.array([i * dL * e_axis for i in range(N_nodes)])
    elements = []
    for i in range(N_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 0.0, 1.0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    e_t = np.array([1.0, -1.0, 0.0])
    e_t /= np.linalg.norm(e_t)
    P = 1000.0
    tip_force = P * e_t
    nodal_loads = {N_nodes - 1: [tip_force[0], tip_force[1], tip_force[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_u = u[6 * (N_nodes - 1):6 * (N_nodes - 1) + 3]
    delta_ref = P * L_total ** 3 / (3.0 * E * I)
    delta_num = float(np.dot(tip_u, e_t))
    assert np.isclose(delta_num, delta_ref, rtol=0.001, atol=1e-09)
    axial_comp = float(np.dot(tip_u, e_axis))
    assert abs(axial_comp) <= delta_ref * 1e-06 + 1e-12
    n_perp = np.cross(e_axis, e_t)
    n_perp /= np.linalg.norm(n_perp)
    out_of_plane_comp = float(np.dot(tip_u, n_perp))
    assert abs(out_of_plane_comp) <= delta_ref * 1e-06 + 1e-12
    support_reaction_force = r[0:3]
    support_reaction_moment = r[3:6]
    assert np.allclose(support_reaction_force + tip_force, np.zeros(3), rtol=1e-09, atol=1e-09)
    r_tip = node_coords[-1] - node_coords[0]
    moment_due_to_force = np.cross(r_tip, tip_force)
    assert np.allclose(support_reaction_moment + moment_due_to_force, np.zeros(3), rtol=1e-09, atol=1e-09)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 1.5], [0.5, 1.5, 1.0], [-0.2, 0.5, 2.0]])
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    E = 70000000000.0
    nu = 0.33
    A = 0.0075
    I = 6e-06
    J = 2.0 * I

    def pick_local_z(vi, vj):
        d = vj - vi
        d /= np.linalg.norm(d)
        ez = np.array([0.0, 0.0, 1.0])
        ey = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(d, ez)) > 0.99:
            return ey
        return ez
    elements = []
    for ni, nj in connections:
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': pick_local_z(node_coords[ni], node_coords[nj]).tolist()})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    loads0 = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, loads0)
    assert np.allclose(u0, 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(r0, 0.0, atol=1e-12, rtol=0.0)
    loads1 = {2: [120.0, -250.0, 180.0, 5.0, 12.0, -8.0], 4: [-300.0, 400.0, -150.0, 0.0, 13.0, -21.0], 5: [50.0, -30.0, 100.0, -3.0, 4.0, 9.0]}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    loads2 = {n: (np.array(v, dtype=float) * 2.0).tolist() for n, v in loads1.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-09, atol=1e-09)
    loads_neg = {n: (np.array(v, dtype=float) * -1.0).tolist() for n, v in loads1.items()}
    u_neg, r_neg = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r_neg, -r1, rtol=1e-09, atol=1e-09)
    F_applied = np.zeros(3)
    M_applied_about_origin = np.zeros(3)
    for n, vec in loads1.items():
        v = np.array(vec, dtype=float)
        F = v[:3]
        M = v[3:]
        F_applied += F
        rpos = node_coords[n]
        M_applied_about_origin += M + np.cross(rpos, F)
    Rf = r1[0:3]
    Rm = r1[3:6]
    assert np.allclose(Rf + F_applied, np.zeros(3), rtol=1e-09, atol=1e-09)
    assert np.allclose(Rm + M_applied_about_origin, np.zeros(3), rtol=1e-09, atol=1e-09)