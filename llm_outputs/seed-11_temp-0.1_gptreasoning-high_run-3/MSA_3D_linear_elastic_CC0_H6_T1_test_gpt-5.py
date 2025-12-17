def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements,
    tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_elems = 10
    n_nodes = n_elems + 1
    L = 2.5
    axis = np.array([1.0, 1.0, 1.0])
    axis_unit = axis / np.linalg.norm(axis)
    node_coords = np.array([i * (L / n_elems) * axis_unit for i in range(n_nodes)])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 6e-06
    J = 2 * I
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1234.5
    load_dir = np.array([1.0, -1.0, 0.0])
    load_dir = load_dir / np.linalg.norm(load_dir)
    assert abs(np.dot(load_dir, axis_unit)) < 1e-12
    tip_force = P * load_dir
    nodal_loads = {n_nodes - 1: [tip_force[0], tip_force[1], tip_force[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert u.shape == (6 * n_nodes,)
    assert r.shape == (6 * n_nodes,)
    assert np.allclose(u[0:6], 0.0, atol=1e-14, rtol=0.0)
    delta_analytical = P * L ** 3 / (3.0 * E * I)
    u_tip = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    delta_numeric = np.linalg.norm(u_tip)
    assert np.isclose(delta_numeric, delta_analytical, rtol=0.001, atol=0.0)
    axial_disp = abs(np.dot(u_tip, axis_unit))
    assert axial_disp <= 1e-06 * delta_numeric + 1e-15
    if delta_numeric > 0:
        cos_angle = abs(np.dot(u_tip / delta_numeric, load_dir))
        assert cos_angle > 1.0 - 0.001
    total_applied_force = np.zeros(3)
    for n, load in nodal_loads.items():
        total_applied_force += np.array(load[:3], dtype=float)
    total_reaction_force = np.zeros(3)
    for i in range(n_nodes):
        total_reaction_force += r[6 * i:6 * i + 3]
    assert np.allclose(total_reaction_force + total_applied_force, 0.0, atol=1e-09, rtol=0.0)
    total_applied_moment = np.zeros(3)
    for n, load in nodal_loads.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:], dtype=float)
        rpos = node_coords[n]
        total_applied_moment += M + np.cross(rpos, F)
    total_reaction_moment = np.zeros(3)
    for i in range(n_nodes):
        RF = r[6 * i:6 * i + 3]
        RM = r[6 * i + 3:6 * i + 6]
        rpos = node_coords[i]
        total_reaction_moment += RM + np.cross(rpos, RF)
    assert np.allclose(total_reaction_moment + total_applied_moment, 0.0, atol=1e-08, rtol=0.0)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 1.5], [1.0, 1.5, 1.5]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.012
    I = 5e-06
    J = 1e-05
    connectivity = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    elements = []
    for ni, nj in connectivity:
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_0 = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, nodal_loads_0)
    assert u0.shape == (6 * n_nodes,)
    assert r0.shape == (6 * n_nodes,)
    assert np.allclose(u0, 0.0, atol=1e-14, rtol=0.0)
    assert np.allclose(r0, 0.0, atol=1e-14, rtol=0.0)
    nodal_loads = {2: [1000.0, -500.0, 200.0, 10.0, -20.0, 30.0], 5: [0.0, 800.0, -400.0, 5.0, 0.0, -15.0], 3: [0.0, 0.0, 0.0, 0.0, 40.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.any(np.abs(u) > 0.0)
    assert np.any(np.abs(r) > 0.0)
    assert np.allclose(u[0:6], 0.0, atol=1e-14, rtol=0.0)
    nodal_loads_2x = {n: [2.0 * v for v in loads] for n, loads in nodal_loads.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, nodal_loads_2x)
    assert np.allclose(u2, 2.0 * u, rtol=1e-09, atol=1e-12)
    assert np.allclose(r2, 2.0 * r, rtol=1e-09, atol=1e-12)
    nodal_loads_neg = {n: [-v for v in loads] for n, loads in nodal_loads.items()}
    u_neg, r_neg = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u_neg, -u, rtol=1e-09, atol=1e-12)
    assert np.allclose(r_neg, -r, rtol=1e-09, atol=1e-12)
    total_applied_force = np.zeros(3)
    total_applied_moment = np.zeros(3)
    for n, load in nodal_loads.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:], dtype=float)
        rpos = node_coords[n]
        total_applied_force += F
        total_applied_moment += M + np.cross(rpos, F)
    total_reaction_force = np.zeros(3)
    total_reaction_moment = np.zeros(3)
    for i in range(n_nodes):
        RF = r[6 * i:6 * i + 3]
        RM = r[6 * i + 3:6 * i + 6]
        rpos = node_coords[i]
        total_reaction_force += RF
        total_reaction_moment += RM + np.cross(rpos, RF)
    assert np.allclose(total_reaction_force + total_applied_force, 0.0, atol=1e-08, rtol=0.0)
    assert np.allclose(total_reaction_moment + total_applied_moment, 0.0, atol=5e-08, rtol=0.0)