def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 3.0
    n_elem = 10
    n_nodes = n_elem + 1
    axis = np.array([1.0, 1.0, 1.0])
    x_dir = axis / np.linalg.norm(axis)
    node_coords = np.array([i * (L / n_elem) * x_dir for i in range(n_nodes)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 8.333e-06
    J = 1.6666e-05
    z_global = np.array([0.0, 0.0, 1.0])
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': z_global})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    y_dir = np.cross(z_global, x_dir)
    y_dir /= np.linalg.norm(y_dir)
    F_mag = 1000.0
    F_vec = F_mag * y_dir
    tip = n_nodes - 1
    nodal_loads = {tip: np.hstack([F_vec, np.zeros(3)])}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u_tip = u[6 * tip:6 * tip + 3]
    delta_num = float(np.dot(u_tip, y_dir))
    delta_ref = F_mag * L ** 3 / (3.0 * E * I)
    assert np.isclose(delta_num, delta_ref, rtol=0.001, atol=1e-12)
    assert np.isclose(np.dot(u_tip, x_dir), 0.0, atol=1e-10)
    r_support_F = r[0:3]
    r_support_M = r[3:6]
    r_tip_vec = node_coords[tip] - node_coords[0]
    total_force = F_vec
    total_moment = np.cross(r_tip_vec, F_vec)
    assert np.allclose(r_support_F + total_force, np.zeros(3), rtol=0, atol=1e-08)
    assert np.allclose(r_support_M + total_moment, np.zeros(3), rtol=0, atol=1e-08)

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
    Lx, Ly, Lz = (2.0, 1.6, 1.3)
    node_coords = np.array([[0.0, 0.0, 0.0], [Lx, 0.0, 0.0], [0.0, Ly, 0.0], [0.0, 0.0, Lz]], dtype=float)
    N = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.015
    I_y = 1.2e-05
    I_z = 9e-06
    J = 2.1e-05
    connections = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)]
    elements = []
    for ni, nj in connections:
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert u0.shape == (6 * N,)
    assert r0.shape == (6 * N,)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads = {1: np.array([1000.0, -2000.0, 500.0, 200.0, 0.0, -100.0], dtype=float), 2: np.array([-800.0, 1000.0, 1200.0, -50.0, 100.0, 0.0], dtype=float), 3: np.array([0.0, -1500.0, -700.0, 0.0, 75.0, 50.0], dtype=float)}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, loads)
    assert not np.allclose(u1, 0.0, atol=1e-12)
    assert not np.allclose(r1, 0.0, atol=1e-12)
    loads2 = {k: 2.0 * v for k, v in loads.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-07, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-07, atol=1e-09)
    loads_neg = {k: -v for k, v in loads.items()}
    u_neg, r_neg = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-07, atol=1e-09)
    assert np.allclose(r_neg, -r1, rtol=1e-07, atol=1e-09)
    total_force = np.zeros(3)
    total_moment = np.zeros(3)
    for n, load in loads.items():
        F = load[:3]
        M = load[3:]
        r_vec = node_coords[n] - node_coords[0]
        total_force += F
        total_moment += M + np.cross(r_vec, F)
    rF = r1[0:3]
    rM = r1[3:6]
    assert np.allclose(rF + total_force, np.zeros(3), rtol=0, atol=1e-07)
    assert np.allclose(rM + total_moment, np.zeros(3), rtol=0, atol=1e-07)