def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 200000000000.0
    nu = 0.3
    n_elements = 10
    P_ref_val = -1.0
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for r in radii:
        for L in lengths:
            I = np.pi * r ** 4 / 4.0
            P_cr_analytical = np.pi ** 2 * E * I / (2.0 * L) ** 2
            n_nodes = n_elements + 1
            node_coords = np.zeros((n_nodes, 3))
            node_coords[:, 2] = np.linspace(0, L, n_nodes)
            A = np.pi * r ** 2
            J = np.pi * r ** 4 / 2.0
            I_rho = I * 2.0
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
            boundary_conditions = {0: [True] * 6}
            nodal_loads = {n_elements: [0.0, 0.0, P_ref_val, 0.0, 0.0, 0.0]}
            (lambda_cr, _) = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
            P_cr_numerical = lambda_cr * abs(P_ref_val)
            assert np.isclose(P_cr_numerical, P_cr_analytical, rtol=1e-05)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 200000000000.0
    nu = 0.3
    L = 15.0
    (b, h) = (0.1, 0.2)
    n_elements = 8
    A = b * h
    Iy = b * h ** 3 / 12.0
    Iz = h * b ** 3 / 12.0
    J = b ** 3 * h * (1 / 3 - 0.21 * (b / h) * (1 - b ** 4 / (12 * h ** 4)))
    I_rho = Iy + Iz
    P_ref_val = -1.0
    n_nodes = n_elements + 1
    node_coords_base = np.zeros((n_nodes, 3))
    node_coords_base[:, 2] = np.linspace(0, L, n_nodes)
    local_z_base = np.array([1.0, 0.0, 0.0])
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_base})
    boundary_conditions = {0: [True] * 6}
    nodal_loads_base = {n_elements: [0.0, 0.0, P_ref_val, 0.0, 0.0, 0.0]}
    (lambda_base, mode_base) = fcn(node_coords=node_coords_base, elements=elements_base, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_base)
    rot_axis = np.array([0.5, -0.3, 0.8])
    rot_axis /= np.linalg.norm(rot_axis)
    R = Rotation.from_rotvec(np.pi / 3 * rot_axis).as_matrix()
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ local_z_base
    load_vec_base = np.array(nodal_loads_base[n_elements])
    load_vec_rot = np.concatenate((R @ load_vec_base[:3], R @ load_vec_base[3:]))
    nodal_loads_rot = {n_elements: load_vec_rot}
    elements_rot = []
    for i in range(n_elements):
        elements_rot.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': local_z_rot})
    (lambda_rot, mode_rot) = fcn(node_coords=node_coords_rot, elements=elements_rot, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads_rot)
    assert np.isclose(lambda_base, lambda_rot, rtol=1e-09)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    mode_base_transformed = T @ mode_base
    norm_rot = np.linalg.norm(mode_rot)
    norm_base_transformed = np.linalg.norm(mode_base_transformed)
    assert norm_rot > 1e-09 and norm_base_transformed > 1e-09
    mode_rot_norm = mode_rot / norm_rot
    mode_base_transformed_norm = mode_base_transformed / norm_base_transformed
    dot_product = np.dot(mode_rot_norm, mode_base_transformed_norm)
    assert np.isclose(abs(dot_product), 1.0, atol=1e-07)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 200000000000.0
    nu = 0.3
    L = 20.0
    r = 0.5
    P_ref_val = -1.0
    I = np.pi * r ** 4 / 4.0
    P_cr_analytical = np.pi ** 2 * E * I / (2.0 * L) ** 2
    mesh_sizes = [2, 4, 8, 16, 32]
    errors = []
    A = np.pi * r ** 2
    J = np.pi * r ** 4 / 2.0
    I_rho = I * 2.0
    for n_elements in mesh_sizes:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        node_coords[:, 2] = np.linspace(0, L, n_nodes)
        elements = []
        for i in range(n_elements):
            elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': [1.0, 0.0, 0.0]})
        boundary_conditions = {0: [True] * 6}
        nodal_loads = {n_elements: [0.0, 0.0, P_ref_val, 0.0, 0.0, 0.0]}
        (lambda_cr, _) = fcn(node_coords=node_coords, elements=elements, boundary_conditions=boundary_conditions, nodal_loads=nodal_loads)
        P_cr_numerical = lambda_cr * abs(P_ref_val)
        relative_error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
        errors.append(relative_error)
    for i in range(len(errors) - 1):
        assert errors[i + 1] < errors[i]
    assert errors[-1] < 1e-07