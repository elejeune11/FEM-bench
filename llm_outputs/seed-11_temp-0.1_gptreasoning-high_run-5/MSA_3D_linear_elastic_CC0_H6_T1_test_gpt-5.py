def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    n_elems = 10
    n_nodes = n_elems + 1
    L0 = 3.0
    end_vec = np.array([L0, L0, L0], dtype=float)
    node_coords = np.linspace(0.0, 1.0, n_nodes)[:, None] * end_vec[None, :]
    E = 210000000000.0
    nu = 0.3
    A = 0.003
    I = 8.333e-06
    J = 2.0 * I
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    axis_dir = end_vec / np.linalg.norm(end_vec)
    load_dir = np.array([1.0, -1.0, 0.0], dtype=float)
    load_dir /= np.linalg.norm(load_dir)
    P = 1000.0
    F = P * load_dir
    nodal_loads = {n_nodes - 1: [F[0], F[1], F[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u = np.asarray(u).ravel()
    r = np.asarray(r).ravel()
    L = np.linalg.norm(end_vec)
    delta_ref = P * L ** 3 / (3.0 * E * I)
    tip_disp = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    assert np.isclose(np.dot(tip_disp, load_dir), delta_ref, rtol=0.002, atol=1e-12)
    assert np.isclose(np.dot(tip_disp, axis_dir), 0.0, atol=max(1e-12, delta_ref * 1e-09))
    r_nodes = r.reshape(n_nodes, 6)
    rF_total = r_nodes[:, :3].sum(axis=0)
    rM_total = r_nodes[:, 3:6].sum(axis=0) + np.cross(node_coords, r_nodes[:, :3]).sum(axis=0)
    F_ext_total = F.copy()
    M_ext_total = np.cross(node_coords[-1], F)
    assert np.allclose(rF_total + F_ext_total, np.zeros(3), rtol=1e-08, atol=1e-08)
    assert np.allclose(rM_total + M_ext_total, np.zeros(3), rtol=1e-08, atol=1e-08)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 1.0, 0.0], [4.0, 1.0, 2.0], [2.0, 0.0, 2.0], [0.0, 0.0, 2.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    conns = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4)]
    E = 100000000000.0
    nu = 0.3
    A = 0.02
    I_y = 0.0004
    I_z = 0.0004
    J = 0.0008
    elements = []
    for i, j in conns:
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1], 2: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u0 = np.asarray(u0).ravel()
    r0 = np.asarray(r0).ravel()
    assert np.allclose(u0, 0.0)
    assert np.allclose(r0, 0.0)
    loads1 = {3: [1200.0, -800.0, 2500.0, 0.0, 1000.0, -500.0], 4: [-600.0, 1400.0, -1000.0, 700.0, 0.0, 0.0]}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, loads1)
    u1 = np.asarray(u1).ravel()
    r1 = np.asarray(r1).ravel()
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    loads2 = {k: [2 * x for x in v] for k, v in loads1.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, loads2)
    u2 = np.asarray(u2).ravel()
    r2 = np.asarray(r2).ravel()
    assert np.allclose(u2, 2.0 * u1, rtol=1e-07, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-07, atol=1e-09)
    loads_neg = {k: [-x for x in v] for k, v in loads1.items()}
    u_neg, r_neg = fcn(node_coords, elements, boundary_conditions, loads_neg)
    u_neg = np.asarray(u_neg).ravel()
    r_neg = np.asarray(r_neg).ravel()
    assert np.allclose(u_neg, -u1, rtol=1e-07, atol=1e-09)
    assert np.allclose(r_neg, -r1, rtol=1e-07, atol=1e-09)
    r_nodes = r1.reshape(n_nodes, 6)
    rF_total = r_nodes[:, :3].sum(axis=0)
    rM_total = r_nodes[:, 3:6].sum(axis=0) + np.cross(node_coords, r_nodes[:, :3]).sum(axis=0)
    F_ext_total = np.zeros(3, dtype=float)
    M_ext_total = np.zeros(3, dtype=float)
    for nid, ld in loads1.items():
        F_i = np.array(ld[:3], dtype=float)
        M_i = np.array(ld[3:6], dtype=float)
        pos_i = node_coords[nid]
        F_ext_total += F_i
        M_ext_total += M_i + np.cross(pos_i, F_i)
    assert np.allclose(rF_total + F_ext_total, np.zeros(3), rtol=1e-06, atol=1e-06)
    assert np.allclose(rM_total + M_ext_total, np.zeros(3), rtol=1e-06, atol=1e-06)