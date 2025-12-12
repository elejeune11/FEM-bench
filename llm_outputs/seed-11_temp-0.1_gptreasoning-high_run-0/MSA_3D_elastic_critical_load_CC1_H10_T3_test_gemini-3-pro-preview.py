def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for L in lengths:
        for r in radii:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4.0
            J = np.pi * r ** 4 / 2.0
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            n_elems = 10
            n_nodes = n_elems + 1
            zs = np.linspace(0, L, n_nodes)
            xs = np.zeros(n_nodes)
            ys = np.zeros(n_nodes)
            node_coords = np.column_stack((xs, ys, zs))
            elements = []
            for i in range(n_elems):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref_mag = 1.0
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref_mag, 0.0, 0.0, 0.0]}
            (lam, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_calc = lam * P_ref_mag
            assert np.isclose(P_calc, P_euler, rtol=0.0001), f'Mismatch for L={L}, r={r}. Calc: {P_calc}, Analytical: {P_euler}'

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
    n_elems = 5
    n_nodes = n_elems + 1
    xs = np.linspace(0, L, n_nodes)
    node_coords_base = np.column_stack((xs, np.zeros(n_nodes), np.zeros(n_nodes)))
    (E, nu) = (200000000000.0, 0.3)
    Iy = 2e-05
    Iz = 0.0001
    J = 0.0001
    A = 0.01
    elements_base = []
    local_z_base = np.array([0.0, 0.0, 1.0])
    for i in range(n_elems):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_base})
    bc_base = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1000.0
    loads_base = {n_nodes - 1: [-P_ref, 0.0, 0.0, 0.0, 0.0, 0.0]}
    (lam_base, mode_base) = fcn(node_coords_base, elements_base, bc_base, loads_base)
    ang_y = np.radians(30)
    ang_z = np.radians(45)
    Ry = np.array([[np.cos(ang_y), 0, np.sin(ang_y)], [0, 1, 0], [-np.sin(ang_y), 0, np.cos(ang_y)]])
    Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0], [np.sin(ang_z), np.cos(ang_z), 0], [0, 0, 1]])
    R = Rz @ Ry
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for el in elements_base:
        new_el = el.copy()
        new_el['local_z'] = local_z_rot
        elements_rot.append(new_el)
    base_load_vec = np.array(loads_base[n_nodes - 1])
    F_rot = R @ base_load_vec[:3]
    M_rot = R @ base_load_vec[3:]
    loads_rot = {n_nodes - 1: np.concatenate((F_rot, M_rot))}
    bc_rot = bc_base
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bc_rot, loads_rot)
    assert np.isclose(lam_base, lam_rot, rtol=1e-05), f'Critical loads differ. Base: {lam_base}, Rotated: {lam_rot}'
    blocks = []
    node_block = np.zeros((6, 6))
    node_block[:3, :3] = R
    node_block[3:, 3:] = R
    for _ in range(n_nodes):
        blocks.append(node_block)
    T = block_diag(*blocks)
    mode_base_transformed = T @ mode_base

    def normalize_mode(v):
        idx = np.argmax(np.abs(v))
        return v / v[idx]
    m1 = normalize_mode(mode_base_transformed)
    m2 = normalize_mode(mode_rot)
    diff = np.linalg.norm(m1 - m2)
    assert diff < 0.0001, f'Mode shapes do not match after rotation transform. Norm diff: {diff}'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    L = 10.0
    r = 0.2
    E = 200000000000.0
    nu = 0.0
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = 2 * I
    P_exact = np.pi ** 2 * E * I / (4 * L ** 2)
    mesh_sizes = [2, 4, 8, 16]
    errors = []
    for n_elems in mesh_sizes:
        n_nodes = n_elems + 1
        zs = np.linspace(0, L, n_nodes)
        node_coords = np.column_stack((np.zeros(n_nodes), np.zeros(n_nodes), zs))
        elements = []
        for i in range(n_elems):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J})
        bc = {0: [1, 1, 1, 1, 1, 1]}
        loads = {n_nodes - 1: [0, 0, -1000.0, 0, 0, 0]}
        (lam, _) = fcn(node_coords, elements, bc, loads)
        P_num = lam * 1000.0
        rel_error = abs(P_num - P_exact) / P_exact
        errors.append(rel_error)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i], f'Error did not decrease. Mesh {mesh_sizes[i]}: {errors[i]}, Mesh {mesh_sizes[i + 1]}: {errors[i + 1]}'
    assert errors[-1] < 0.001, f'Finest mesh error is not low enough: {errors[-1]}'