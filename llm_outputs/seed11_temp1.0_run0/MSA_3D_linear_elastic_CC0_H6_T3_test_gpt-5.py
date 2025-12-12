def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    L = 3.0
    n_elems = 10
    n_nodes = n_elems + 1
    axis_vec = np.array([1.0, 1.0, 1.0])
    axis_unit = axis_vec / np.linalg.norm(axis_vec)
    node_coords = np.array([i * (L / n_elems) * axis_unit for i in range(n_nodes)])
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 8e-06
    J = 2.0 * I
    elements = [{'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J} for i in range(n_elems)]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    F_dir = np.array([1.0, -1.0, 0.0])
    F_hat = F_dir / np.linalg.norm(F_dir)
    F = P * F_hat
    tip = n_nodes - 1
    nodal_loads = {tip: [F[0], F[1], F[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_analytical = P * L ** 3 / (3.0 * E * I)
    u_tip = np.array(u[6 * tip:6 * tip + 3])
    u_along_load = float(np.dot(u_tip, F_hat))
    assert np.isclose(u_along_load, delta_analytical, rtol=1e-05, atol=1e-09)
    u_along_axis = float(np.dot(u_tip, axis_unit))
    assert np.isclose(u_along_axis, 0.0, atol=max(1e-12, 1e-08 * abs(delta_analytical)))
    total_reaction_force = np.zeros(3)
    for i in range(n_nodes):
        total_reaction_force += np.array(r[6 * i:6 * i + 3])
    assert np.allclose(total_reaction_force, -F, rtol=1e-09, atol=1e-06)

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
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [3.0, 2.0, 1.5], [3.0, 3.0, 3.0], [4.0, 1.0, 2.5]])
    n_nodes = node_coords.shape[0]
    E = 70000000000.0
    nu = 0.33
    A = 0.01
    I = 1e-05
    J = 2e-05
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 1, 'node_j': 4, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}, {'node_i': 4, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J}]
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    loads0 = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, loads0)
    assert np.allclose(u0, np.zeros_like(u0), atol=1e-12)
    assert np.allclose(r0, np.zeros_like(r0), atol=1e-12)
    loads1 = {2: [0.0, 0.0, 0.0, 50.0, -25.0, 10.0], 3: [5000.0, -2000.0, 1000.0, 100.0, -50.0, 20.0], 4: [-1000.0, 2000.0, -500.0, -10.0, 20.0, -30.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1) > 0.0
    loads2 = {k: list(2.0 * np.array(v, dtype=float)) for (k, v) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-09, atol=1e-09)
    loads3 = {k: list(-np.array(v, dtype=float)) for (k, v) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -u1, rtol=1e-09, atol=1e-09)
    assert np.allclose(r3, -r1, rtol=1e-09, atol=1e-09)
    F_applied = np.zeros(3)
    M_applied = np.zeros(3)
    for (n, load) in loads1.items():
        F_n = np.array(load[:3], dtype=float)
        M_n = np.array(load[3:], dtype=float)
        r_pos = node_coords[n]
        F_applied += F_n
        M_applied += M_n + np.cross(r_pos, F_n)
    F_reac = np.zeros(3)
    M_reac = np.zeros(3)
    for i in range(n_nodes):
        Rf_i = np.array(r1[6 * i:6 * i + 3], dtype=float)
        Rm_i = np.array(r1[6 * i + 3:6 * i + 6], dtype=float)
        r_pos = node_coords[i]
        F_reac += Rf_i
        M_reac += Rm_i + np.cross(r_pos, Rf_i)
    assert np.allclose(F_reac + F_applied, np.zeros(3), rtol=1e-09, atol=1e-06)
    assert np.allclose(M_reac + M_applied, np.zeros(3), rtol=1e-09, atol=1e-05)