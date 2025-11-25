def test_simple_beam_discretized_axis_111(fcn):
    """Verify cantilever beam along [1,1,1] axis with tip load against analytical solution"""
    L = np.sqrt(3)
    n_elements = 10
    dx = L / n_elements / np.sqrt(3)
    nodes = np.array([[i * dx, i * dx, i * dx] for i in range(n_elements + 1)])
    E = 200000000000.0
    nu = 0.3
    d = 0.05
    A = np.pi * d ** 2 / 4
    I = np.pi * d ** 4 / 64
    J = 2 * I
    elements = []
    for i in range(n_elements):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2)])})
    F = 1000.0
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    F_proj = F / np.sqrt(2)
    nodal_loads = {n_elements: [0, -F_proj, F_proj, 0, 0, 0]}
    (u, r) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    delta_analytical = F * L ** 3 / (3 * E * I)
    tip_y = u[6 * n_elements + 1]
    tip_z = u[6 * n_elements + 2]
    delta_numerical = np.sqrt(tip_y ** 2 + tip_z ** 2)
    assert_allclose(delta_numerical, delta_analytical, rtol=0.01)

def test_complex_geometry_and_basic_loading(fcn):
    """Test 3D frame analysis with various loading conditions"""
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': None}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u0, r0) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    assert_allclose(u0, np.zeros_like(u0))
    assert_allclose(r0, np.zeros_like(r0))
    F = 1000.0
    M = 1000.0
    nodal_loads = {2: [F, 0, F, M, 0, M]}
    (u1, r1) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    assert not np.allclose(u1, np.zeros_like(u1))
    nodal_loads = {2: [2 * F, 0, 2 * F, 2 * M, 0, 2 * M]}
    (u2, r2) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    assert_allclose(u2, 2 * u1, rtol=1e-10)
    assert_allclose(r2, 2 * r1, rtol=1e-10)
    nodal_loads = {2: [-F, 0, -F, -M, 0, -M]}
    (u3, r3) = fcn(nodes, elements, boundary_conditions, nodal_loads)
    assert_allclose(u3, -u1, rtol=1e-10)
    assert_allclose(r3, -r1, rtol=1e-10)
    (u, r) = fcn(nodes, elements, boundary_conditions, {2: [F, 0, F, M, 0, M]})
    sum_fx = np.sum(r[0::6])
    sum_fy = np.sum(r[1::6])
    sum_fz = np.sum(r[2::6])
    assert_allclose([sum_fx, sum_fy, sum_fz], [-F, 0, -F], rtol=1e-10)
    sum_mx = np.sum(r[3::6] + nodes[:, 1] * r[2::6] - nodes[:, 2] * r[1::6])
    sum_my = np.sum(r[4::6] + nodes[:, 2] * r[0::6] - nodes[:, 0] * r[2::6])
    sum_mz = np.sum(r[5::6] + nodes[:, 0] * r[1::6] - nodes[:, 1] * r[0::6])
    assert_allclose([sum_mx, sum_my, sum_mz], [-M, 0, -M], rtol=1e-10)