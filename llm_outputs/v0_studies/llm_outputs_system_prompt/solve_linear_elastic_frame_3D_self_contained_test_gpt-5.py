def test_simple_beam_discretized_axis_111(fcn):
    """Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution."""
    np = fcn.__globals__['np']
    n_el = 10
    L = 3.0
    axis = np.array([1.0, 1.0, 1.0])
    n_hat = axis / np.linalg.norm(axis)
    N = n_el + 1
    node_coords = np.array([i * (L / n_el) * n_hat for i in range(N)], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-05
    elements = []
    for i in range(n_el):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0.0, 0.0, 1.0])})
    boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
    p_dir = np.array([1.0, -1.0, 0.0])
    p_hat = p_dir / np.linalg.norm(p_dir)
    P = 1000.0
    nodal_loads = {n_el: [*P * p_hat, 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    tip = n_el
    u_tip = np.array(u[tip * 6:tip * 6 + 3])
    delta_num = float(np.dot(u_tip, p_hat))
    delta_ref = P * L ** 3 / (3 * E * I)
    assert np.isclose(delta_num, delta_ref, rtol=0.01, atol=0.0)

def test_complex_geometry_and_basic_loading(fcn):
    """Test linear 3D frame analysis on a non-trivial geometry under various loading conditions.
    Suggested Test stages:
    1. Zero loads -> All displacements and reactions should be zero.
    2. Apply mixed forces and moments at free nodes -> Displacements and reactions should be nonzero.
    3. Double the loads -> Displacements and reactions should double (linearity check).
    4. Negate the original loads -> Displacements and reactions should flip sign.
    5. Assure that reactions (forces and moments) satisfy global static equilibrium"""
    np = fcn.__globals__['np']
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)
    E = 70000000000.0
    nu = 0.3
    A = 0.02
    I = 2e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0.0, 1.0, 0.0])}]
    boundary_conditions = {0: (1, 1, 1, 1, 1, 1)}
    nodal_loads_0 = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads_0)
    assert np.allclose(u0, 0.0) and np.allclose(r0, 0.0)
    loads1 = {2: [10.0, -20.0, 30.0, 5.0, 0.0, -3.0], 3: [-15.0, 5.0, -10.0, 2.0, -4.0, 1.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.any(np.abs(u1) > 0.0)
    assert np.any(np.abs(r1) > 0.0)
    loads2 = {k: [2 * v for v in vals] for (k, vals) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * np.array(u1))
    assert np.allclose(r2, 2.0 * np.array(r1))
    loads3 = {k: [-v for v in vals] for (k, vals) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -np.array(u1))
    assert np.allclose(r3, -np.array(r1))
    total_F = np.zeros(3)
    total_M = np.zeros(3)
    for (n, vals) in loads1.items():
        F = np.array(vals[:3], dtype=float)
        M = np.array(vals[3:], dtype=float)
        rpos = node_coords[n]
        total_F += F
        total_M += M + np.cross(rpos, F)
    r1 = np.array(r1, dtype=float)
    N = len(node_coords)
    reac_F = np.zeros(3)
    reac_M = np.zeros(3)
    for i in range(N):
        dof0 = 6 * i
        reac_F += r1[dof0:dof0 + 3]
        reac_M += r1[dof0 + 3:dof0 + 6]
    assert np.allclose(reac_F + total_F, 0.0, atol=1e-08)
    assert np.allclose(reac_M + total_M, 0.0, atol=1e-08)

def test_ill_conditioned_due_to_under_constrained_structure(fcn):
    """Test that solve_linear_elastic_frame_3d raises a ValueError
    when the structure is improperly constrained, leading to an
    ill-conditioned free-free stiffness matrix (K_ff).
    The solver should detect this (cond(K_ff) >= 1e16) and raise a ValueError."""
    np = fcn.__globals__['np']
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': 2 * I, 'local_z': np.array([0.0, 0.0, 1.0])}]
    boundary_conditions = {}
    nodal_loads = {}
    raised = False
    try:
        fcn(node_coords, elements, boundary_conditions, nodal_loads)
    except ValueError:
        raised = True
    assert raised