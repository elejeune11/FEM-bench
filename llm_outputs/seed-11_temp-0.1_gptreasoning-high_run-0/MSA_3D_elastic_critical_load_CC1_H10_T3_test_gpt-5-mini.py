def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler
    cantilever analytical solution. Use 10 elements and tolerances appropriate
    for anticipated discretization error (~1e-5).
    """
    import numpy as np
    import math
    E = 210000000000.0
    nu = 0.3
    radii = (0.5, 0.75, 1.0)
    lengths = (10.0, 20.0, 40.0)
    n_elems = 10
    tol_rel = 1e-05
    for r in radii:
        A = math.pi * r ** 2
        I = math.pi * r ** 4 / 4.0
        J = math.pi * r ** 4 / 2.0
        for L in lengths:
            n_nodes = n_elems + 1
            coords = np.zeros((n_nodes, 3), dtype=float)
            coords[:, 2] = np.linspace(0.0, L, n_nodes)
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': int(i), 'node_j': int(i + 1), 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(I), 'I_z': float(I), 'J': float(J)})
            bcs = {0: (1, 1, 1, 1, 1, 1)}
            nodal_loads = {n_nodes - 1: (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)}
            (lam, mode) = fcn(coords, elements, bcs, nodal_loads)
            mode = np.asarray(mode)
            assert float(lam) > 0.0
            p_cr_analytic = math.pi ** 2 * E * I / (4.0 * L ** 2)
            rel_err = abs(float(lam) - p_cr_analytic) / p_cr_analytic
            assert rel_err < tol_rel
            assert mode.shape == (6 * n_nodes,)
            assert np.allclose(mode[0:6], 0.0, atol=1e-08)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """Orientation invariance test with a rectangular section (Iy ≠ Iz).
    Solve the cantilever in base orientation and after a rigid-body rotation R
    applied to node coordinates, element local axes, and applied loads. The
    critical load factor λ must match and the rotated mode must equal T @ mode_base
    up to scale and sign (T block-diagonal applies R to translational and
    rotational DOFs at each node).
    """
    import numpy as np
    import math

    def rodrigues(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        return math.cos(theta) * np.eye(3) + (1.0 - math.cos(theta)) * np.outer(axis, axis) + math.sin(theta) * K
    E = 1000000.0
    nu = 0.3
    b = 1.0
    h = 2.0
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = 0.2 * A * min(b, h)
    L = 20.0
    n_elems = 10
    n_nodes = n_elems + 1
    coords = np.zeros((n_nodes, 3), dtype=float)
    coords[:, 2] = np.linspace(0.0, L, n_nodes)
    local_z_base = np.array([1.0, 0.0, 0.0], dtype=float)
    elements_base = []
    for i in range(n_elems):
        elements_base.append({'node_i': int(i), 'node_j': int(i + 1), 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(I_y), 'I_z': float(I_z), 'J': float(J), 'local_z': local_z_base.copy()})
    bcs = {0: (1, 1, 1, 1, 1, 1)}
    nodal_loads_base = {n_nodes - 1: (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)}
    (lam_base, mode_base) = fcn(coords, elements_base, bcs, nodal_loads_base)
    mode_base = np.asarray(mode_base)
    assert float(lam_base) > 0.0
    axis = np.array([1.0, 1.0, 0.5], dtype=float)
    theta = 0.7
    R = rodrigues(axis, theta)
    coords_rot = (R @ coords.T).T
    elements_rot = []
    for el in elements_base:
        lz = np.asarray(el.get('local_z', local_z_base), dtype=float)
        lz_rot = (R @ lz).tolist()
        el_rot = dict(el)
        el_rot['local_z'] = lz_rot
        elements_rot.append(el_rot)
    nodal_loads_rot = {}
    for (nid, load) in nodal_loads_base.items():
        F = np.asarray(load[0:3], dtype=float)
        M = np.asarray(load[3:6], dtype=float)
        F_rot = (R @ F).tolist()
        M_rot = (R @ M).tolist()
        nodal_loads_rot[int(nid)] = tuple(list(F_rot) + list(M_rot))
    (lam_rot, mode_rot) = fcn(coords_rot, elements_rot, bcs, nodal_loads_rot)
    mode_rot = np.asarray(mode_rot)
    assert float(lam_rot) > 0.0
    rel_err_lambda = abs(float(lam_rot) - float(lam_base)) / max(abs(float(lam_base)), 1.0)
    assert rel_err_lambda < 1e-09
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    x = T @ mode_base
    denom = float(np.dot(x, x))
    assert denom > 0.0
    s = float(np.dot(x, mode_rot) / denom)
    residual = np.linalg.norm(mode_rot - s * x) / (np.linalg.norm(mode_rot) + 1e-18)
    assert residual < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and check that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that
    the finest mesh achieves very high accuracy.
    """
    import numpy as np
    import math
    E = 21000000000.0
    nu = 0.3
    r = 1.0
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    J = math.pi * r ** 4 / 2.0
    L = 10.0
    meshes = (4, 8, 16, 32)
    rel_errors = []
    for n_elems in meshes:
        n_nodes = n_elems + 1
        coords = np.zeros((n_nodes, 3), dtype=float)
        coords[:, 2] = np.linspace(0.0, L, n_nodes)
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': int(i), 'node_j': int(i + 1), 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(I), 'I_z': float(I), 'J': float(J)})
        bcs = {0: (1, 1, 1, 1, 1, 1)}
        nodal_loads = {n_nodes - 1: (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)}
        (lam, mode) = fcn(coords, elements, bcs, nodal_loads)
        lam = float(lam)
        assert lam > 0.0
        p_cr_analytic = math.pi ** 2 * E * I / (4.0 * L ** 2)
        rel_err = abs(lam - p_cr_analytic) / p_cr_analytic
        rel_errors.append(rel_err)
    for i in range(1, len(rel_errors)):
        assert rel_errors[i] <= rel_errors[i - 1] + 1e-12
    assert rel_errors[-1] < 1e-06