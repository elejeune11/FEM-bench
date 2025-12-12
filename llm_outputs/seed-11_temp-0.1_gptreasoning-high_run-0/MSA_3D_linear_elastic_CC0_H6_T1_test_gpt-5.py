def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    n_elems = 10
    L_total = 3.0
    axis = np.array([1.0, 1.0, 1.0])
    axis_unit = axis / np.linalg.norm(axis)
    step = L_total / n_elems * axis_unit
    node_coords = np.array([i * step for i in range(n_elems + 1)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 8.333e-06
    J = 1.4e-05
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    v_perp = np.cross(axis_unit, np.array([0.0, 0.0, 1.0]))
    v_perp_norm = np.linalg.norm(v_perp)
    assert v_perp_norm > 0.0
    v_perp = v_perp / v_perp_norm
    F_mag = 1000.0
    F_global = F_mag * v_perp
    nodal_loads = {n_elems: [F_global[0], F_global[1], F_global[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    u_tip = u[6 * n_elems:6 * n_elems + 3]
    delta_analytical = F_mag * L_total ** 3 / (3.0 * E * I)
    delta_numeric_along_load = float(np.dot(u_tip, v_perp))
    assert np.isclose(delta_numeric_along_load, delta_analytical, rtol=0.01, atol=1e-10)
    u_tip_perp = u_tip - delta_numeric_along_load * v_perp
    assert np.linalg.norm(u_tip_perp) <= 0.001 * abs(delta_analytical)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 1.0], [1.0, 1.0, 2.0], [3.0, 1.0, 1.0]], dtype=float)
    E = 70000000000.0
    nu = 0.3
    A = 0.02
    I = 1.8e-05
    J = 3.6e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 3, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 4, 'node_j': 5, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 5, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 3, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 2, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u0, 0.0, atol=1e-14, rtol=0.0)
    assert np.allclose(r0, 0.0, atol=1e-14, rtol=0.0)
    nodal_loads = {2: [3000.0, -2000.0, 1000.0, 100.0, 50.0, -25.0], 4: [-500.0, 800.0, -1200.0, 0.0, -60.0, 10.0], 5: [0.0, -1000.0, 500.0, 20.0, 0.0, 40.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.linalg.norm(u1) > 0.0
    support_reaction_norm = np.linalg.norm(r1[0:3]) + np.linalg.norm(r1[3:6])
    assert support_reaction_norm > 0.0
    nodal_loads_double = {n: [2.0 * x for x in load] for (n, load) in nodal_loads.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_double)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-08, atol=1e-10)
    nodal_loads_neg = {n: [-x for x in load] for (n, load) in nodal_loads.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_neg)
    assert np.allclose(u3, -u1, rtol=1e-08, atol=1e-10)
    assert np.allclose(r3, -r1, rtol=1e-08, atol=1e-10)
    n_nodes = node_coords.shape[0]
    sum_F_applied = np.zeros(3)
    sum_M_applied = np.zeros(3)
    for (nid, load) in nodal_loads.items():
        F = np.array(load[:3], dtype=float)
        M = np.array(load[3:], dtype=float)
        rpos = node_coords[nid]
        sum_F_applied += F
        sum_M_applied += M + np.cross(rpos, F)
    sum_F_react = np.zeros(3)
    sum_M_react = np.zeros(3)
    for nid in range(n_nodes):
        Fi = np.array(r1[6 * nid:6 * nid + 3], dtype=float)
        Mi = np.array(r1[6 * nid + 3:6 * nid + 6], dtype=float)
        rpos = node_coords[nid]
        sum_F_react += Fi
        sum_M_react += Mi + np.cross(rpos, Fi)
    assert np.allclose(sum_F_react + sum_F_applied, np.zeros(3), rtol=1e-08, atol=1e-06)
    assert np.allclose(sum_M_react + sum_M_applied, np.zeros(3), rtol=1e-08, atol=1e-06)