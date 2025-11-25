def test_simple_beam_discretized_axis_111(fcn):
    """
    Verification with resepct to a known analytical solution.
    Cantilever beam, axis along [1,1,1]. Ten equal 3-D frame elements, tip loaded by a transverse force
    perpendicular to the beam axis. Verify beam tip deflection with the appropriate analytical reference solution.
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    r_sec = 0.02
    A = np.pi * r_sec ** 2
    I = np.pi * r_sec ** 4 / 4.0
    J = 2.0 * I
    L = 3.0
    N = 11
    d = np.array([1.0, 1.0, 1.0])
    d = d / np.linalg.norm(d)
    s = np.linspace(0.0, L, N)
    node_coords = (s[:, None] * d[None, :]).astype(float)
    elements = []
    z_ref = np.array([0.0, 0.0, 1.0])
    y_ref = np.array([0.0, 1.0, 0.0])
    for i in range(N - 1):
        axis = node_coords[i + 1] - node_coords[i]
        axis = axis / np.linalg.norm(axis)
        local_z = z_ref if abs(np.dot(axis, z_ref)) < 0.99 else y_ref
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P = 1000.0
    t = np.cross(d, np.array([1.0, 0.0, 0.0]))
    t = t / np.linalg.norm(t)
    F_tip = P * t
    nodal_loads = {N - 1: [F_tip[0], F_tip[1], F_tip[2], 0.0, 0.0, 0.0]}
    (u, r) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
    delta_expected = P * L ** 3 / (3.0 * E * I)
    u_tip = u[6 * (N - 1):6 * (N - 1) + 3]
    delta_numeric = float(np.dot(u_tip, t))
    assert np.isclose(delta_numeric, delta_expected, rtol=0.02, atol=0.0)
    n_dof = 6 * N
    fixed_mask = np.zeros(n_dof, dtype=bool)
    for i in range(N):
        bc = boundary_conditions.get(i, [0, 0, 0, 0, 0, 0])
        for j in range(6):
            if bc[j] == 1:
                fixed_mask[6 * i + j] = True
    free_mask = ~fixed_mask
    assert np.allclose(r[free_mask], 0.0, atol=1e-08 * P)
    R_total_force = np.zeros(3)
    for i in range(N):
        R_total_force += r[6 * i:6 * i + 3]
    F_applied_total = np.zeros(3)
    for (k, vals) in nodal_loads.items():
        F_applied_total += np.asarray(vals[:3], dtype=float)
    assert np.allclose(R_total_force, -F_applied_total, rtol=1e-09, atol=1e-09 * max(1.0, np.linalg.norm(F_applied_total)))

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
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 1.0, 0.0], [1.2, 1.0, 0.8], [2.0, 1.0, 0.8]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 70000000000.0
    nu = 0.29
    A = 0.0008
    I_y = 5e-06
    I_z = 4e-06
    J = I_y + I_z
    conn = [(0, 1), (1, 2), (2, 3), (3, 4)]
    z_ref = np.array([0.0, 0.0, 1.0])
    y_ref = np.array([0.0, 1.0, 0.0])
    elements = []
    for (ni, nj) in conn:
        axis = node_coords[nj] - node_coords[ni]
        axis = axis / np.linalg.norm(axis)
        local_z = z_ref if abs(np.dot(axis, z_ref)) < 0.99 else y_ref
        elements.append({'node_i': ni, 'node_j': nj, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}

    def fixed_free_masks(n):
        fixed_mask = np.zeros(6 * n, dtype=bool)
        for i in range(n):
            bc = boundary_conditions.get(i, [0, 0, 0, 0, 0, 0])
            for j in range(6):
                if bc[j] == 1:
                    fixed_mask[6 * i + j] = True
        return (fixed_mask, ~fixed_mask)
    (fixed_mask, free_mask) = fixed_free_masks(n_nodes)
    nodal_loads_0 = {}
    (u0, r0) = fcn(node_coords, elements, boundary_conditions, nodal_loads_0)
    assert np.allclose(u0, 0.0, atol=1e-14)
    assert np.allclose(r0, 0.0, atol=1e-14)
    nodal_loads_1 = {4: [50.0, -30.0, 20.0, 5.0, -2.0, 3.0], 3: [10.0, 0.0, -40.0, 0.0, 8.0, -4.0]}
    (u1, r1) = fcn(node_coords, elements, boundary_conditions, nodal_loads_1)
    assert np.linalg.norm(u1) > 0.0
    assert np.linalg.norm(r1[fixed_mask]) > 0.0
    assert np.allclose(r1[free_mask], 0.0, atol=1e-10)
    nodal_loads_2 = {k: [2.0 * v for v in vals] for (k, vals) in nodal_loads_1.items()}
    (u2, r2) = fcn(node_coords, elements, boundary_conditions, nodal_loads_2)
    assert np.allclose(u2, 2.0 * u1, rtol=1e-09, atol=1e-12)
    assert np.allclose(r2, 2.0 * r1, rtol=1e-09, atol=1e-12)
    nodal_loads_n = {k: [-v for v in vals] for (k, vals) in nodal_loads_1.items()}
    (un, rn) = fcn(node_coords, elements, boundary_conditions, nodal_loads_n)
    assert np.allclose(un, -u1, rtol=1e-09, atol=1e-12)
    assert np.allclose(rn, -r1, rtol=1e-09, atol=1e-12)
    F_applied = np.zeros(3)
    M_applied = np.zeros(3)
    for (k, vals) in nodal_loads_1.items():
        F_applied += np.asarray(vals[:3], dtype=float)
        M_applied += np.asarray(vals[3:6], dtype=float)
    M_from_forces = np.zeros(3)
    for (k, vals) in nodal_loads_1.items():
        rpos = node_coords[k]
        Fk = np.asarray(vals[:3], dtype=float)
        M_from_forces += np.cross(rpos, Fk)
    R_total_force = np.zeros(3)
    R_total_moment = np.zeros(3)
    for i in range(n_nodes):
        R_total_force += r1[6 * i:6 * i + 3]
        R_total_moment += r1[6 * i + 3:6 * i + 6]
    scaleF = max(1.0, np.linalg.norm(F_applied))
    scaleM = max(1.0, np.linalg.norm(M_applied) + np.linalg.norm(M_from_forces))
    assert np.allclose(R_total_force, -F_applied, rtol=1e-09, atol=1e-09 * scaleF)
    assert np.allclose(R_total_moment, -(M_applied + M_from_forces), rtol=1e-09, atol=1e-09 * scaleM)