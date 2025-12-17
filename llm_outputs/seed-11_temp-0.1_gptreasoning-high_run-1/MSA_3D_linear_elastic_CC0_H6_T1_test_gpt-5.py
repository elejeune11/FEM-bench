def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 3.0
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    n_elem = 10
    n_nodes = n_elem + 1
    s_vals = np.linspace(0.0, L, n_nodes)
    node_coords = np.array([axis * s for s in s_vals])
    E = 210000000000.0
    nu = 0.3
    A = 0.005
    I = 8e-06
    J = 2.0 * I
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 0.0, 1.0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    load_dir = np.array([-1.0, 1.0, 0.0])
    load_dir = load_dir / np.linalg.norm(load_dir)
    F_mag = 1000.0
    F_global = F_mag * load_dir
    nodal_loads = {n_nodes - 1: [F_global[0], F_global[1], F_global[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
    tip_u = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    tip_disp_along_load = float(np.dot(tip_u, load_dir))
    w_expected = F_mag * L ** 3 / (3.0 * E * I)
    assert np.isclose(tip_disp_along_load, w_expected, rtol=0.0001, atol=1e-09)
    assert np.allclose(u[:6], 0.0, atol=1e-12)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [2.0, 1.5, 0.5], [1.0, 1.5, 1.5], [0.5, 0.0, 2.0]])
    n_nodes = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.008
    I = 6e-06
    J = 2.0 * I
    elements = []
    conn = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for ni, nj in conn:
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': [0.0, 0.0, 1.0]})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_0 = {}
    u0, r0 = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_0)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    nodal_loads_1 = {2: [15000.0, -5000.0, 12000.0, 300.0, -200.0, 150.0], 4: [-7000.0, 9000.0, -3000.0, -100.0, 250.0, -50.0]}
    u1, r1 = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_1)
    assert np.linalg.norm(u1) > 0.0
    assert np.any(np.abs(r1) > 0.0)
    nodal_loads_2 = {k: [2.0 * x for x in v] for k, v in nodal_loads_1.items()}
    u2, r2 = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-10, atol=1e-10)
    nodal_loads_3 = {k: [-x for x in v] for k, v in nodal_loads_1.items()}
    u3, r3 = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_3)
    assert np.allclose(u3, -u1, rtol=1e-10, atol=1e-10)
    assert np.allclose(r3, -r1, rtol=1e-10, atol=1e-10)
    F_ext = np.zeros(3)
    M_ext = np.zeros(3)
    for nidx, load in nodal_loads_1.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:], dtype=float)
        rpos = node_coords[nidx]
        F_ext += F
        M_ext += M + np.cross(rpos, F)
    F_react = np.zeros(3)
    M_react = np.zeros(3)
    for nidx, bc in boundary_conditions.items():
        ri = r1[6 * nidx:6 * nidx + 6]
        F_i = ri[:3]
        M_i = ri[3:]
        rpos = node_coords[nidx]
        F_react += F_i
        M_react += M_i + np.cross(rpos, F_i)
    F_res = F_react + F_ext
    M_res = M_react + M_ext
    F_scale = max(1.0, np.linalg.norm(F_ext))
    M_scale = max(1.0, np.linalg.norm(M_ext))
    assert np.linalg.norm(F_res) <= 1e-09 * F_scale
    assert np.linalg.norm(M_res) <= 1e-09 * M_scale