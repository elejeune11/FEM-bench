def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.001
    I = 1e-06
    J = 2 * I
    L = 1.0
    n_elems = 10
    n_nodes = n_elems + 1
    axis = np.array([1.0, 1.0, 1.0])
    axis /= np.linalg.norm(axis)
    dx = L / n_elems
    node_coords = np.array([i * dx * axis for i in range(n_nodes)], dtype=float)
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    v = np.array([-1.0, 1.0, 0.0])
    v -= axis * np.dot(v, axis)
    v /= np.linalg.norm(v)
    Fmag = 1234.5
    Fvec = Fmag * v
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: np.array([Fvec[0], Fvec[1], Fvec[2], 0.0, 0.0, 0.0])}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip = n_nodes - 1
    tip_u = u[6 * tip:6 * tip + 3]
    delta_analytical = Fmag * L ** 3 / (3.0 * E * I)
    delta_numerical = float(np.dot(tip_u, v))
    assert np.isclose(delta_numerical, delta_analytical, rtol=0.01, atol=1e-10)
    axial_comp = float(np.dot(tip_u, axis))
    assert abs(axial_comp) <= 1e-06 * max(1.0, abs(delta_analytical))

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
    E = 210000000000.0
    nu = 0.3
    A = 0.001
    I = 4e-06
    J = 2 * I
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([1.0, 0.0, 0.0])}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u0, 0.0)
    assert np.allclose(r0, 0.0)
    loads1 = {2: np.array([1000.0, -500.0, 200.0, 10.0, 20.0, -30.0]), 3: np.array([0.0, 800.0, -400.0, 5.0, -15.0, 25.0])}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    for nid in [1, 2, 3]:
        assert np.allclose(r1[6 * nid:6 * nid + 6], 0.0, atol=1e-10)
    loads2 = {k: 2.0 * v for (k, v) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-12, atol=1e-12)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-12, atol=1e-12)
    loads3 = {k: -1.0 * v for (k, v) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -u1, rtol=1e-12, atol=1e-12)
    assert np.allclose(r3, -r1, rtol=1e-12, atol=1e-12)
    applied_F = np.zeros(3)
    applied_M = np.zeros(3)
    for (nid, load) in loads1.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:6], dtype=float)
        x = node_coords[nid]
        applied_F += F
        applied_M += M + np.cross(x, F)
    react_F = np.zeros(3)
    react_M = np.zeros(3)
    N = node_coords.shape[0]
    for nid in range(N):
        rf = r1[6 * nid:6 * nid + 3]
        rm = r1[6 * nid + 3:6 * nid + 6]
        x = node_coords[nid]
        react_F += rf
        react_M += rm + np.cross(x, rf)
    tolF = 1e-08 * max(1.0, np.linalg.norm(applied_F))
    tolM = 1e-08 * max(1.0, np.linalg.norm(applied_M))
    assert np.linalg.norm(applied_F + react_F) <= tolF
    assert np.linalg.norm(applied_M + react_M) <= tolM

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """
    Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError.
    """
    E = 70000000000.0
    nu = 0.33
    A = 0.002
    I = 5e-06
    J = 2 * I
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
    boundary_conditions = {}
    nodal_loads = {}
    with pytest.raises(ValueError):
        fcn(node_coords, elements, boundary_conditions, nodal_loads)