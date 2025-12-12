def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    n_elems = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    P_ref_mag = 1000.0
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4.0
            J = np.pi * r ** 4 / 2.0
            node_coords = np.zeros((n_elems + 1, 3))
            node_coords[:, 2] = np.linspace(0, L, n_elems + 1)
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            nodal_loads = {n_elems: [0.0, 0.0, -P_ref_mag, 0.0, 0.0, 0.0]}
            (lam, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_critical_calc = lam * P_ref_mag
            P_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
            assert np.isclose(P_critical_calc, P_analytical, rtol=0.001), f'Failed for r={r}, L={L}: Calc={P_critical_calc}, Analytical={P_analytical}'

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    L = 10.0
    n_elems = 6
    (E, nu) = (210000000000.0, 0.3)
    (b, h) = (0.2, 0.5)
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = 0.1 * (Iy + Iz)
    node_coords_base = np.zeros((n_elems + 1, 3))
    node_coords_base[:, 0] = np.linspace(0, L, n_elems + 1)
    local_z_base = np.array([0.0, 0.0, 1.0])
    elements_base = []
    for i in range(n_elems):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_base})
    bc_base = {0: [1, 1, 1, 1, 1, 1]}
    P_mag = 10000.0
    loads_base = {n_elems: [-P_mag, 0.0, 0.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, bc_base, loads_base)
    theta = np.radians(30)
    phi = np.radians(45)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    R = Rz @ Ry
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for el in elements_base:
        new_el = el.copy()
        new_el['local_z'] = local_z_rot
        elements_rot.append(new_el)
    F_base_vec = np.array(loads_base[n_elems][:3])
    F_rot_vec = R @ F_base_vec
    loads_rot = {n_elems: [F_rot_vec[0], F_rot_vec[1], F_rot_vec[2], 0.0, 0.0, 0.0]}
    bc_rot = bc_base
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc_rot, loads_rot)
    assert np.isclose(lam_base, lam_rot, rtol=1e-05), f'Critical load factor mismatch: Base={lam_base}, Rot={lam_rot}'
    n_nodes = n_elems + 1
    ndof = 6 * n_nodes
    T = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        idx = i * 6
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    expected_mode_rot = T @ mode_base
    norm_calc = np.linalg.norm(mode_rot)
    norm_exp = np.linalg.norm(expected_mode_rot)
    assert norm_calc > 1e-09 and norm_exp > 1e-09
    dot_product = np.dot(mode_rot, expected_mode_rot)
    similarity = abs(dot_product / (norm_calc * norm_exp))
    assert np.isclose(similarity, 1.0, atol=0.001), f'Mode shapes do not match after rotation transformation. Similarity: {similarity}'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 20.0
    r = 0.5
    E = 200000000000.0
    nu = 0.3
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    P_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    P_ref = 1000.0
    mesh_sizes = [2, 4, 8, 16]
    errors = []
    for n in mesh_sizes:
        node_coords = np.zeros((n + 1, 3))
        node_coords[:, 2] = np.linspace(0, L, n + 1)
        elements = []
        for i in range(n):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': None})
        bc = {0: [1, 1, 1, 1, 1, 1]}
        loads = {n: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lam, _) = fcn(node_coords, elements, bc, loads)
        P_calc = lam * P_ref
        rel_error = abs(P_calc - P_analytical) / P_analytical
        errors.append(rel_error)
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], f'Error did not decrease from {mesh_sizes[i - 1]} to {mesh_sizes[i]} elements. Errors: {errors}'
    assert errors[-1] < 0.001, f'Finest mesh error too high: {errors[-1]}'