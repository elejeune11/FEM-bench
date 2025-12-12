def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case apply a unit compressive tip load (Fz = -1). With E = 1 and
    a unit reference load the returned elastic critical load factor λ should
    match the analytical Euler cantilever value P_cr = π^2 E I / (4 L^2)
    within a small relative tolerance appropriate for ~10 elements.
    """
    import numpy as np
    import math
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    n_elems = 10
    E = 1.0
    nu = 0.3
    rtol = 0.0001
    for r in radii:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in lengths:
            n_nodes = n_elems + 1
            z = np.linspace(0.0, L, n_nodes)
            node_coords = np.vstack([np.zeros_like(z), np.zeros_like(z), z]).T
            elements = []
            local_z = np.array([1.0, 0.0, 0.0])
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z})
            bcs = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, bcs, nodal_loads)
            lam = float(np.real(lam))
            assert lam > 0.0
            assert mode.shape == (6 * n_nodes,)
            P_cr_analytic = math.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(lam - P_cr_analytic) / abs(P_cr_analytic)
            assert rel_err <= rtol, f'r={r}, L={L}, rel_err={rel_err}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever in base orientation and again after applying a rigid-body
    rotation R to node coordinates, element local_z vectors, and nodal loads.
    The critical load factor λ must be identical for both. The buckling mode
    from the rotated model should equal T @ mode_base (allowing for scale/sign),
    where T applies R to translational and rotational DOFs at each node.
    """
    import numpy as np

    def rodrigues(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    n_elems = 10
    E = 1.0
    nu = 0.3
    L = 20.0
    I_y = 2.0
    I_z = 0.7
    A = 1.0
    J = 0.5
    n_nodes = n_elems + 1
    z = np.linspace(0.0, L, n_nodes)
    node_coords = np.vstack([np.zeros_like(z), np.zeros_like(z), z]).T
    local_z = np.array([1.0, 0.0, 0.0])
    elements = []
    for i in range(n_elems):
        elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
    bcs = {0: [1, 1, 1, 1, 1, 1]}
    nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords, elements, bcs, nodal_loads)
    lam_base = float(np.real(lam_base))
    mode_base = np.asarray(mode_base, dtype=float)
    axis = np.array([1.0, 1.0, 0.3])
    theta = 0.4321
    R = rodrigues(axis, theta)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for el in elements:
        el_rot = el.copy()
        el_rot['local_z'] = (R @ np.asarray(el['local_z'])).tolist()
        elements_rot.append(el_rot)
    nodal_loads_rot = {}
    for (idx, loads) in nodal_loads.items():
        f = np.asarray(loads[:3], dtype=float)
        m = np.asarray(loads[3:], dtype=float)
        f_rot = R @ f
        m_rot = R @ m
        nodal_loads_rot[idx] = np.concatenate([f_rot, m_rot]).tolist()
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bcs, nodal_loads_rot)
    lam_rot = float(np.real(lam_rot))
    mode_rot = np.asarray(mode_rot, dtype=float)
    assert lam_base > 0.0 and lam_rot > 0.0
    assert np.isclose(lam_base, lam_rot, rtol=1e-09, atol=1e-12)
    transformed = np.zeros_like(mode_base)
    for i in range(n_nodes):
        u = mode_base[6 * i:6 * i + 3]
        th = mode_base[6 * i + 3:6 * i + 6]
        transformed[6 * i:6 * i + 3] = R @ u
        transformed[6 * i + 3:6 * i + 6] = R @ th
    norm_t = np.linalg.norm(transformed)
    norm_r = np.linalg.norm(mode_rot)
    assert norm_t > 0 and norm_r > 0
    t_normed = transformed / norm_t
    r_normed = mode_rot / norm_r
    cond1 = np.allclose(t_normed, r_normed, rtol=1e-06, atol=1e-08)
    cond2 = np.allclose(t_normed, -r_normed, rtol=1e-06, atol=1e-08)
    assert cond1 or cond2

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical
    load approaches the analytical Euler value with decreasing relative error,
    and that the finest mesh achieves very high accuracy.
    """
    import numpy as np
    import math
    radi = 1.0
    L = 20.0
    E = 1.0
    nu = 0.3
    A = math.pi * radi ** 2
    I = math.pi * radi ** 4 / 4.0
    J = math.pi * radi ** 4 / 2.0
    n_elems_list = [4, 8, 16, 32]
    errors = []
    for n_elems in n_elems_list:
        n_nodes = n_elems + 1
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.vstack([np.zeros_like(z), np.zeros_like(z), z]).T
        elements = []
        local_z = np.array([1.0, 0.0, 0.0])
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z})
        bcs = {0: [1, 1, 1, 1, 1, 1]}
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]}
        (lam, mode) = fcn(node_coords, elements, bcs, nodal_loads)
        lam = float(np.real(lam))
        P_cr_analytic = math.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(lam - P_cr_analytic) / abs(P_cr_analytic)
        errors.append(rel_err)
    assert errors[-1] < errors[0]
    for (a, b) in zip(errors, errors[1:]):
        assert b <= a + 1e-08
    assert errors[-1] < 1e-06