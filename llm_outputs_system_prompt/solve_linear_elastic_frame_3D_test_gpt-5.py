def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force perpendicular to the beam axis.
    Verify beam tip deflection with the appropriate analytical reference solution.
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    I = 8e-06
    J = 0.001
    L = 3.0
    n_el = 10
    n_nodes = n_el + 1
    axis = np.array([1.0, 1.0, 1.0])
    x_unit = axis / np.linalg.norm(axis)
    load_dir = np.array([1.0, -1.0, 0.0])
    load_dir = load_dir - np.dot(load_dir, x_unit) * x_unit
    load_dir = load_dir / np.linalg.norm(load_dir)
    z_local = np.cross(x_unit, load_dir)
    z_local = z_local / np.linalg.norm(z_local)
    node_coords = np.zeros((n_nodes, 3), dtype=float)
    for i in range(n_nodes):
        s = L * i / n_el
        node_coords[i, :] = x_unit * s
    elements = []
    for e in range(n_el):
        elements.append({'node_i': e, 'node_j': e + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': z_local})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    F_tip = P * load_dir
    nodal_loads = {n_nodes - 1: [F_tip[0], F_tip[1], F_tip[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    assert isinstance(u, np.ndarray) and isinstance(r, np.ndarray)
    assert u.shape == (6 * n_nodes,)
    assert r.shape == (6 * n_nodes,)
    tip_u = u[6 * (n_nodes - 1):6 * (n_nodes - 1) + 3]
    proj = float(np.dot(tip_u, load_dir))
    delta_expected = P * L ** 3 / (3.0 * E * I)
    assert abs(proj - delta_expected) <= 1e-05 * abs(delta_expected) + 1e-12
    ortho = tip_u - proj * load_dir
    assert np.linalg.norm(ortho) <= max(1e-08 * abs(delta_expected), 1e-10)
    support_reaction_force = r[:3]
    assert np.allclose(support_reaction_force, -F_tip, atol=1e-07 * np.linalg.norm(F_tip) + 1e-12)

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
    import numpy as np
    E = 70000000000.0
    nu = 0.33
    A = 0.01
    I = 5e-06
    J = 0.001
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.5, 0.5]], dtype=float)
    n_nodes = node_coords.shape[0]

    def local_z_for(i, j):
        x = node_coords[j] - node_coords[i]
        x = x / np.linalg.norm(x)
        g = np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(np.cross(x, g)) < 1e-08:
            g = np.array([0.0, 1.0, 0.0])
        z = np.cross(x, g)
        z = z / np.linalg.norm(z)
        return z
    connectivity = [(0, 1), (1, 2), (1, 3), (3, 4), (2, 4)]
    elements = []
    for (i, j) in connectivity:
        elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z_for(i, j)})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    loads0 = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, loads0)
    assert np.allclose(u0, 0.0, atol=1e-14)
    assert np.allclose(r0, 0.0, atol=1e-14)
    loads1 = {2: [10.0, 20.0, -30.0, 1.0, -2.0, 3.0], 4: [-5.0, 0.0, 15.0, 0.0, 1.0, 0.0], 3: [0.0, 0.0, 0.0, -1.5, 0.5, 0.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, loads1)
    assert np.max(np.abs(u1)) > 1e-12
    tol_scale = 1.0 + sum((abs(val) for ld in loads1.values() for val in ld))
    assert np.max(np.abs(r1[6:])) <= 1e-08 * tol_scale + 1e-12
    F_sum = np.zeros(3)
    M_sum = np.zeros(3)
    for (n, ld) in loads1.items():
        F = np.array(ld[:3], dtype=float)
        M = np.array(ld[3:], dtype=float)
        pos = node_coords[n]
        F_sum += F
        M_sum += M + np.cross(pos, F)
    support_force = r1[:3]
    support_moment = r1[3:6]
    assert np.allclose(support_force, -F_sum, atol=1e-08 * np.linalg.norm(F_sum) + 1e-10)
    assert np.allclose(support_moment, -M_sum, atol=1e-08 * np.linalg.norm(M_sum) + 1e-10)
    loads2 = {n: [2.0 * v for v in ld] for (n, ld) in loads1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, loads2)
    assert np.allclose(u2, 2.0 * u1, atol=1e-09 * np.max(np.abs(u1)) + 1e-12)
    assert np.allclose(r2, 2.0 * r1, atol=1e-09 * np.max(np.abs(r1)) + 1e-12)
    loads3 = {n: [-v for v in ld] for (n, ld) in loads1.items()}
    (u3, r3) = fcn(node_coords, elements, boundary_conditions, loads3)
    assert np.allclose(u3, -u1, atol=1e-09 * np.max(np.abs(u1)) + 1e-12)
    assert np.allclose(r3, -r1, atol=1e-09 * np.max(np.abs(r1)) + 1e-12)