def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    radii = [0.5, 0.75, 1.0]
    lengths = [10, 20, 40]
    n_elements = 10
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4
            J = np.pi * r ** 4 / 2
            I_rho = I
            E = 210000000000.0
            nu = 0.3
            n_nodes = n_elements + 1
            node_coords = np.zeros((n_nodes, 3))
            for i in range(n_nodes):
                node_coords[i, 2] = i * L / n_elements
            elements = []
            for i in range(n_elements):
                elements.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': I, 'Iz': I, 'J': J, 'I_rho': I_rho, 'local_z': None})
            boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
            nodal_loads = {n_nodes - 1: [0, 0, -1.0, 0, 0, 0]}
            (critical_load_factor, _) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_euler_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
            P_cr_numerical = critical_load_factor * 1.0
            assert_allclose(P_cr_numerical, P_euler_analytical, rtol=0.0001)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.  
    The buckling mode from the rotated model should equal the base mode transformed by R:
    [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    (b, h) = (0.2, 0.4)
    A = b * h
    Iy = b * h ** 3 / 12
    Iz = h * b ** 3 / 12
    J = b * h ** 3 * (1 / 3 - 0.21 * h / b * (1 - h ** 4 / (12 * b ** 4)))
    I_rho = Iy + Iz
    E = 210000000000.0
    nu = 0.3
    L = 10.0
    n_elements = 5
    n_nodes = n_elements + 1
    node_coords_base = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords_base[i, 2] = i * L / n_elements
    elements_base = []
    for i in range(n_elements):
        elements_base.append({'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'I_rho': I_rho, 'local_z': [0, 0, 1]})
    boundary_conditions = {0: [0, 1, 2, 3, 4, 5]}
    nodal_loads_base = {n_nodes - 1: [0, 0, -1.0, 0, 0, 0]}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    theta = np.pi / 4
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    node_coords_rot = node_coords_base @ R.T
    elements_rot = []
    for ele in elements_base:
        ele_rot = ele.copy()
        if ele['local_z'] is not None:
            ele_rot['local_z'] = list(R @ np.array(ele['local_z']))
        elements_rot.append(ele_rot)
    nodal_loads_rot = {}
    for (node, load) in nodal_loads_base.items():
        F = np.array(load[:3])
        M = np.array(load[3:])
        F_rot = R @ F
        M_rot = R @ M
        nodal_loads_rot[node] = list(F_rot) + list(M_rot)
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    assert_allclose(lambda_rot, lambda_base, rtol=1e-10)
    T_full = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        start_idx = 6 * i
        end_idx = 6 * i + 6
        T_full[start_idx:end_idx, start_idx:end_idx] = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    mode_base_transformed = T_full @ mode_base
    scale_factor = np.linalg.norm(mode_rot) / np.linalg.norm(mode_base_transformed)
    if np.dot(mode_rot, mode_base_transformed) < 0:
        scale_factor *= -1
    assert_allclose(mode_rot, scale_factor * mode_base_transformed, rtol=1e-08)

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    r = 0.5
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4
    J = np.pi * r ** 4 / 2
    I_rho = I
    E = 210000000000.0
    nu = 0.3
    L = 10.0
    element_counts = [2, 4, 8, 16, 32]
    errors = []
    P_euler_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    for n_elements in element_counts:
        n