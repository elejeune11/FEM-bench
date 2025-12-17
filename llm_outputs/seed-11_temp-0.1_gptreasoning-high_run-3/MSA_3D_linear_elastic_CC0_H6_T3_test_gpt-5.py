def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L_total = 3.0
    n_el = 10
    n_nodes = n_el + 1
    axis = np.array([1.0, 1.0, 1.0])
    axis_unit = axis / np.linalg.norm(axis)
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    for i in range(n_nodes):
        s = L_total * (i / n_el)
        node_coords[i] = axis_unit * s
    E = 210000000000.0
    nu = 0.3
    A = 0.005
    I = 1e-06
    J = 2.0 * I
    elements = []
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    load_dir = np.array([1.0, -1.0, 0.0])
    load_unit = load_dir / np.linalg.norm(load_dir)
    tip_force = P * load_unit
    nodal_loads = {n_nodes - 1: [tip_force[0], tip_force[1], tip_force[2], 0.0, 0.0, 0.0]}
    u, r = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_disp = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    delta_num = float(np.dot(tip_disp, load_unit))
    axial_disp = float(np.dot(tip_disp, axis_unit))
    assert abs(axial_disp) <= 1e-08 * (1.0 + abs(delta_num))
    delta_ref = P * L_total ** 3 / (3.0 * E * I)
    assert np.isclose(delta_num, delta_ref, rtol=0.001, atol=1e-09)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.4], [2.0, 0.5, 0.1], [1.3, 1.3, 0.8], [2.3, 1.8, 1.2]], dtype=float)
    N = node_coords.shape[0]
    E = 70000000000.0
    nu = 0.33
    A = 0.004
    I = 5e-06
    J = 2.0 * I
    conns = [(0, 1), (1, 2), (1, 3), (0, 3), (2, 3), (2, 4), (3, 4)]
    elements = [{'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J} for i, j in conns]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {}
    u0, r0 = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert np.allclose(u0, 0.0, atol=1e-12)
    assert np.allclose(r0, 0.0, atol=1e-12)
    loads_base = {2: [10.0, -5.0, 25.0, 0.0, 40.0, -15.0], 4: [-20.0, 30.0, 5.0, 10.0, -20.0, 35.0]}
    u1, r1 = fcn(node_coords, elements, boundary_conditions, loads_base)
    assert np.linalg.norm(u1) > 0.0
    assert not np.allclose(r1, 0.0, atol=1e-12)
    loads_double = {k: (np.array(v, dtype=float) * 2.0).tolist() for k, v in loads_base.items()}
    u2, r2 = fcn(node_coords, elements, boundary_conditions, loads_double)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-09, atol=1e-12)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-09, atol=1e-12)
    loads_neg = {k: (np.array(v, dtype=float) * -1.0).tolist() for k, v in loads_base.items()}
    u3, r3 = fcn(node_coords, elements, boundary_conditions, loads_neg)
    assert np.allclose(u3, -u1, rtol=1e-09, atol=1e-12)
    assert np.allclose(r3, -r1, rtol=1e-09, atol=1e-12)
    F_applied = np.zeros(3, dtype=float)
    M_applied = np.zeros(3, dtype=float)
    for nid, load in loads_base.items():
        load_vec = np.asarray(load, dtype=float)
        F_applied += load_vec[:3]
        M_applied += load_vec[3:6]
    F_react = np.zeros(3, dtype=float)
    M_react = np.zeros(3, dtype=float)
    for i in range(N):
        dof0 = 6 * i
        F_react += r1[dof0:dof0 + 3]
        M_react += r1[dof0 + 3:dof0 + 6]
    assert np.allclose(F_react + F_applied, 0.0, atol=1e-06)
    M_react_total = M_react.copy()
    for i in range(N):
        rf = r1[6 * i:6 * i + 3]
        M_react_total += np.cross(node_coords[i], rf)
    M_applied_total = M_applied.copy()
    for nid, load in loads_base.items():
        fv = np.asarray(load[:3], dtype=float)
        M_applied_total += np.cross(node_coords[nid], fv)
    assert np.allclose(M_react_total + M_applied_total, 0.0, atol=1e-06)