def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force
    perpendicular to the beam axis. Verify beam tip deflection with the appropriate analytical reference solution."""
    L = 3.0
    n_el = 10
    n_nodes = n_el + 1
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    node_coords = np.array([i * (L / n_el) * axis for i in range(n_nodes)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    r = 0.05
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = 2.0 * I
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J} for i in range(n_el)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    fhat = np.array([-1.0, 1.0, 0.0], dtype=float)
    fhat /= np.linalg.norm(fhat)
    Fmag = 10000.0
    F_vec = Fmag * fhat
    nodal_loads = {n_nodes - 1: [F_vec[0], F_vec[1], F_vec[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_theory = Fmag * L ** 3 / (3.0 * E * I)
    u_tip = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    delta_model = float(np.dot(u_tip, fhat))
    assert np.isclose(delta_model, delta_theory, rtol=0.01, atol=1e-09), 'Tip deflection mismatch with analytical solution'

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions satisfy global static equilibrium."""
    Lx, Ly, Lz = (3.0, 2.0, 4.0)
    node_coords = np.array([[0.0, 0.0, 0.0], [Lx, 0.0, 0.0], [Lx, Ly, 0.0], [0.0, Ly, 0.0], [0.0, 0.0, Lz], [Lx, 0.0, Lz], [Lx, Ly, Lz], [0.0, Ly, Lz]], dtype=float)
    N = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 8e-06
    J = 2.0 * I
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
    elements = [{'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J} for i, j in edges]
    fixed_nodes = [0, 1, 2, 3]
    boundary_conditions = {n: [1, 1, 1, 1, 1, 1] for n in fixed_nodes}
    nodal_loads = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u0, 0.0, atol=1e-12), 'Displacements should be zero for zero loads'
    assert np.allclose(r0, 0.0, atol=1e-12), 'Reactions should be zero for zero loads'
    loads2 = {4: [1000.0, 0.0, -500.0, 100.0, 50.0, 0.0], 5: [0.0, 2000.0, 0.0, 0.0, -100.0, 80.0], 6: [-1500.0, 0.0, -200.0, 40.0, 0.0, 60.0], 7: [200.0, -300.0, 400.0, 0.0, 0.0, -70.0]}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.linalg.norm(u1) > 0.0, 'Displacements should be nonzero under loads'
    assert np.linalg.norm(r1) > 0.0, 'Reactions should be nonzero under loads'
    for n in range(N):
        r_node = r1[6 * n:6 * n + 6]
        if n in fixed_nodes:
            assert np.linalg.norm(r_node) > 0.0, 'Support reactions should be present at fixed nodes'
        else:
            assert np.allclose(r_node, 0.0, atol=1e-08), 'Reactions should be zero at free nodes'
    F_ext = np.zeros(3)
    M_ext = np.zeros(3)
    for n, load in loads2.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:6], dtype=float)
        pos = node_coords[n]
        F_ext += F
        M_ext += M + np.cross(pos, F)
    F_react = np.zeros(3)
    M_react = np.zeros(3)
    for n in fixed_nodes:
        rF = r1[6 * n:6 * n + 3]
        rM = r1[6 * n + 3:6 * n + 6]
        pos = node_coords[n]
        F_react += rF
        M_react += rM + np.cross(pos, rF)
    assert np.allclose(F_ext + F_react, np.zeros(3), atol=1e-06, rtol=1e-10), 'Global force equilibrium not satisfied'
    assert np.allclose(M_ext + M_react, np.zeros(3), atol=1e-06, rtol=1e-10), 'Global moment equilibrium not satisfied'
    loads2_double = {n: [2.0 * x for x in vals] for n, vals in loads2.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, loads2_double)
    assert np.allclose(u2, 2.0 * u1, atol=1e-08, rtol=1e-10), 'Displacements should double with doubled loads'
    assert np.allclose(r2, 2.0 * r1, atol=1e-08, rtol=1e-10), 'Reactions should double with doubled loads'
    loads2_neg = {n: [-x for x in vals] for n, vals in loads2.items()}
    u3, r3 = fcn(node_coords, elements, boundary_conditions, loads2_neg)
    assert np.allclose(u3, -u1, atol=1e-08, rtol=1e-10), 'Displacements should flip sign with negated loads'
    assert np.allclose(r3, -r1, atol=1e-08, rtol=1e-10), 'Reactions should flip sign with negated loads'