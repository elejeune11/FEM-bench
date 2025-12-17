def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_el = 10
    n_nodes = n_el + 1
    L = 3.0
    v_axis = np.array([1.0, 1.0, 1.0])
    v_axis = v_axis / np.linalg.norm(v_axis)
    t = np.linspace(0.0, L, n_nodes)
    node_coords = t[:, None] * v_axis[None, :]
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 8e-06
    Iy = I
    Iz = I
    J = 2 * I
    elements = []
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    z = np.array([0.0, 0.0, 1.0])
    d = np.cross(v_axis, z)
    d = d / np.linalg.norm(d)
    P = 1000.0
    F_tip = P * d
    nodal_loads = {n_nodes - 1: [F_tip[0], F_tip[1], F_tip[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u_tip_vec = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    u_tip_along_d = float(np.dot(u_tip_vec, d))
    delta = P * L ** 3 / (3.0 * E * I)
    assert np.isclose(u_tip_along_d, delta, rtol=0.02, atol=1e-09)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.2, 1.0, 0.7], [0.2, 1.0, 1.5], [1.5, 0.2, 1.1]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 70000000000.0
    nu = 0.29
    A = 0.003
    Iy = 4e-06
    Iz = 4e-06
    J = 8e-06

    def choose_local_z(i, j):
        vec = node_coords[j] - node_coords[i]
        vec = vec / np.linalg.norm(vec)
        z = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(vec, z)) > 0.98:
            return np.array([0.0, 1.0, 0.0])
        return z
    connections = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)]
    elements = []
    for ni, nj in connections:
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': choose_local_z(ni, nj)})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_zero = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, nodal_loads_zero)
    assert u0.shape == (6 * n_nodes,)
    assert r0.shape == (6 * n_nodes,)
    assert np.allclose(u0, 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(r0, 0.0, atol=1e-12, rtol=0.0)
    loads1 = {1: [1500.0, -700.0, 300.0, 25.0, -12.0, 40.0], 2: [-200.0, 1800.0, -900.0, -15.0, 60.0, -35.0], 3: [-800.0, 0.0, 1300.0, -45.0, 0.0, 20.0], 4: [120.0, -240.0, 0.0, 9.0, 18.0, -7.0]}

    def scale_loads(loads_dict, factor):
        return {k: (np.array(v, dtype=float) * factor).tolist() for k, v in loads_dict.items()}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.linalg.norm(u1[6:]) > 0.0
    assert np.linalg.norm(r1) > 0.0
    loads2 = scale_loads(loads1, 2.0)
    u2, r2 = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-08, atol=1e-10)
    loads_neg = scale_loads(loads1, -1.0)
    u_neg, r_neg = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u_neg, -u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r_neg, -r1, rtol=1e-08, atol=1e-10)
    sum_forces = np.zeros(3)
    sum_moments = np.zeros(3)
    for nd, load in loads1.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:], dtype=float)
        rpos = node_coords[nd]
        sum_forces += F
        sum_moments += np.cross(rpos, F) + M
    r_node0 = r1[0:6]
    rF = r_node0[:3]
    rM = r_node0[3:]
    assert np.allclose(rF, -sum_forces, rtol=1e-08, atol=1e-06)
    assert np.allclose(rM, -sum_moments, rtol=1e-08, atol=1e-06)