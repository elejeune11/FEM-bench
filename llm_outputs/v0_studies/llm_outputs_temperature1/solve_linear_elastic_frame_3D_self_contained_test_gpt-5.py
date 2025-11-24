def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 3.0
    axis = np.array([1.0, 1.0, 1.0])
    xhat = axis / np.linalg.norm(axis)
    Nseg = 10
    nodes = np.array([i * (L / Nseg) * xhat for i in range(Nseg + 1)])
    E = 210000000000.0
    nu = 0.3
    r = 0.05
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = 2.0 * I
    elements = []
    for i in range(Nseg):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    bc = {0: [1, 1, 1, 1, 1, 1]}
    tip = Nseg
    v_perp = np.array([1.0, -1.0, 0.0])
    vhat = v_perp / np.linalg.norm(v_perp)
    P = 1000.0
    loads = {tip: [*P * vhat, 0.0, 0.0, 0.0]}
    (u, r) = fcn(nodes, elements, bc, loads)
    u_tip = u[6 * tip:6 * tip + 3]
    disp_along_load = float(np.dot(u_tip, vhat))
    delta_ref = P * L ** 3 / (3.0 * E * I)
    assert np.isclose(disp_along_load, delta_ref, rtol=0.001, atol=1e-08)
    disp_along_axis = float(np.dot(u_tip, xhat))
    assert np.isclose(disp_along_axis, 0.0, atol=1e-06)

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
    nodes = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, L, 0.0], [0.0, L, 0.0], [0.0, 0.0, L], [L, 0.0, L], [L, L, L], [0.0, L, L]])
    E = 210000000000.0
    nu = 0.3
    r = 0.03
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = 2.0 * I

    def choose_local_z(p_i, p_j):
        v = p_j - p_i
        vhat = v / np.linalg.norm(v)
        z_cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(vhat, z_cand)) > 0.99:
            return np.array([0.0, 1.0, 0.0])
        return z_cand
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    elements = []
    for (i, j) in edges:
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': choose_local_z(nodes[i], nodes[j])})
    bc = {0: [1, 1, 1, 1, 1, 1]}
    loads0 = {}
    (u0, r0) = fcn(nodes, elements, bc, loads0)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads = {2: [1000.0, -500.0, 0.0, 0.0, 0.0, 10.0], 5: [0.0, 0.0, 800.0, 20.0, 0.0, 0.0], 6: [-300.0, 200.0, -400.0, 0.0, -15.0, 5.0], 7: [50.0, 0.0, 0.0, 0.0, 25.0, -30.0]}
    (u1, r1) = fcn(nodes, elements, bc, loads)
    assert np.linalg.norm(u1) > 0.0
    assert np.any(np.abs(r1) > 0.0)
    F_ext = np.zeros(3)
    M_ext = np.zeros(3)
    for (nid, load) in loads.items():
        f = np.array(load[:3], dtype=float)
        m = np.array(load[3:], dtype=float)
        rpos = nodes[nid]
        F_ext += f
        M_ext += m + np.cross(rpos, f)
    F_reac = np.zeros(3)
    M_reac = np.zeros(3)
    N = nodes.shape[0]
    for nid in range(N):
        rf = r1[6 * nid:6 * nid + 3]
        rm = r1[6 * nid + 3:6 * nid + 6]
        rpos = nodes[nid]
        F_reac += rf
        M_reac += rm + np.cross(rpos, rf)
    assert np.allclose(F_ext + F_reac, 0.0, atol=1e-06, rtol=1e-10)
    assert np.allclose(M_ext + M_reac, 0.0, atol=1e-06, rtol=1e-10)
    loads2 = {k: [2.0 * x for x in v] for (k, v) in loads.items()}
    (u2, r2) = fcn(nodes, elements, bc, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-10, atol=1e-10)
    loads_neg = {k: [-x for x in v] for (k, v) in loads.items()}
    (u_neg, r_neg) = fcn(nodes, elements, bc, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r_neg, -r1, rtol=1e-10, atol=1e-10)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """
    Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError.
    """
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    E = 210000000000.0
    nu = 0.3
    A = 0.0001
    I = 1e-08
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2.0 * I, 'local_z': np.array([0.0, 0.0, 1.0])}]
    bc = {}
    loads = {1: [100.0, 50.0, -25.0, 0.0, 0.0, 10.0]}
    with pytest.raises(ValueError):
        fcn(nodes, elements, bc, loads)