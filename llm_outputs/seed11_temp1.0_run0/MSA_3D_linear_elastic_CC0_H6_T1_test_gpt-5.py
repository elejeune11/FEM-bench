def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with respect to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.001
    I = 8e-06
    J = 1e-05
    n_elem = 10
    n_nodes = n_elem + 1
    L_total = 3.0
    axis = np.array([1.0, 1.0, 1.0])
    axis_unit = axis / np.linalg.norm(axis)
    node_coords = np.outer(np.linspace(0.0, L_total, n_nodes), axis_unit)
    elements = []
    for i in range(n_elem):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    w = np.array([1.0, -1.0, 0.0])
    w = w / np.linalg.norm(w)
    F = 1000.0
    Fvec = F * w
    tip = n_nodes - 1
    nodal_loads = {tip: [Fvec[0], Fvec[1], Fvec[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip_u = u[6 * tip:6 * tip + 3]
    defl_along_w = float(np.dot(tip_u, w))
    expected_defl = F * L_total ** 3 / (3.0 * E * I)
    assert np.isclose(defl_along_w, expected_defl, rtol=0.005, atol=1e-09)
    ortho_comp = np.linalg.norm(tip_u - defl_along_w * w)
    assert ortho_comp <= max(1e-09, 1e-06 * expected_defl)

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
    E = 70000000000.0
    nu = 0.33
    A = 0.0008
    Iy = 3e-06
    Iz = 3e-06
    J = 6e-06
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.5, 0.0], [0.5, 2.0, 0.7], [0.2, 0.4, 2.0]], dtype=float)
    connections = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)]
    elements = []
    for (i, j) in connections:
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads_0 = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads_0)
    assert np.allclose(u0, 0.0)
    assert np.allclose(r0, 0.0)
    loads_stage = {1: np.array([1000.0, -500.0, 200.0, 50.0, -20.0, 10.0]), 2: np.array([-300.0, 400.0, -600.0, 0.0, 30.0, -10.0]), 3: np.array([100.0, 200.0, 300.0, 10.0, 0.0, -20.0])}
    nodal_loads_1 = {k: v.tolist() for (k, v) in loads_stage.items()}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    nodal_loads_2 = {k: (2.0 * v).tolist() for (k, v) in loads_stage.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-09, atol=1e-09)
    nodal_loads_3 = {k: (-1.0 * v).tolist() for (k, v) in loads_stage.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, nodal_loads_3)
    assert np.allclose(u3, -u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r3, -r1, rtol=1e-09, atol=1e-09)
    n_nodes = node_coords.shape[0]
    fixed_nodes = [idx for idx in range(n_nodes) if boundary_conditions.get(idx, [0] * 6).count(1) > 0]
    total_applied_force = np.zeros(3)
    total_applied_moment_origin = np.zeros(3)
    for (node_idx, load) in loads_stage.items():
        f = load[:3]
        m = load[3:]
        rpos = node_coords[node_idx]
        total_applied_force += f
        total_applied_moment_origin += m + np.cross(rpos, f)
    total_reaction_force = np.zeros(3)
    total_reaction_moment_origin = np.zeros(3)
    for idx in fixed_nodes:
        rf = r1[6 * idx:6 * idx + 3]
        rm = r1[6 * idx + 3:6 * idx + 6]
        rpos = node_coords[idx]
        total_reaction_force += rf
        total_reaction_moment_origin += rm + np.cross(rpos, rf)
    assert np.allclose(total_reaction_force + total_applied_force, np.zeros(3), rtol=1e-08, atol=1e-06)
    assert np.allclose(total_reaction_moment_origin + total_applied_moment_origin, np.zeros(3), rtol=1e-08, atol=1e-06)