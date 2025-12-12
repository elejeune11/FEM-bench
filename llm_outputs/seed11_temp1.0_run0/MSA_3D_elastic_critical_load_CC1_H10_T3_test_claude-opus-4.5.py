def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 210000.0
    nu = 0.3
    n_elements = 10
    radii = [0.5, 0.75, 1.0]
    lengths = [10.0, 20.0, 40.0]
    for r in radii:
        for L in lengths:
            A = np.pi * r ** 2
            I = np.pi * r ** 4 / 4.0
            J = np.pi * r ** 4 / 2.0
            P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
            n_nodes = n_elements + 1
            node_coords = np.zeros((n_nodes, 3))
            for i in range(n_nodes):
                node_coords[i, 2] = i * L / n_elements
            elements = []
            for i in range(n_elements):
                elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])}
                elements.append(elem)
            boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
            P_ref = 1.0
            nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
            (lambda_cr, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
            P_cr_numerical = lambda_cr * P_ref
            rel_error = abs(P_cr_numerical - P_euler) / P_euler
            assert rel_error < 1e-05, f'Failed for r={r}, L={L}: P_cr_numerical={P_cr_numerical}, P_euler={P_euler}, rel_error={rel_error}'
            assert mode.shape == (6 * n_nodes,)
            assert lambda_cr > 0

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.
    The buckling mode from the rotated model should equal the base mode transformed by R:
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    """
    E = 210000.0
    nu = 0.3
    L = 20.0
    n_elements = 8
    b = 1.0
    h = 2.0
    A = b * h
    I_y = b * h ** 3 / 12.0
    I_z = h * b ** 3 / 12.0
    J = b * h * (b ** 2 + h ** 2) / 12.0
    n_nodes = n_elements + 1
    node_coords_base = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        node_coords_base[i, 2] = i * L / n_elements
    local_z_base = np.array([0.0, 1.0, 0.0])
    elements_base = []
    for i in range(n_elements):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_base.copy()}
        elements_base.append(elem)
    boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
    P_ref = 1.0
    load_base = np.array([0.0, 0.0, -P_ref, 0.0, 0.0, 0.0])
    nodal_loads_base = {n_nodes - 1: load_base.tolist()}
    (lambda_base, mode_base) = fcn(node_coords_base, elements_base, boundary_conditions, nodal_loads_base)
    theta = np.pi / 5.0
    axis = np.array([1.0, 1.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    node_coords_rot = (R @ node_coords_base.T).T
    local_z_rot = R @ local_z_base
    elements_rot = []
    for i in range(n_elements):
        elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot.copy()}
        elements_rot.append(elem)
    load_rot = np.zeros(6)
    load_rot[:3] = R @ load_base[:3]
    load_rot[3:] = R @ load_base[3:]
    nodal_loads_rot = {n_nodes - 1: load_rot.tolist()}
    (lambda_rot, mode_rot) = fcn(node_coords_rot, elements_rot, boundary_conditions, nodal_loads_rot)
    rel_error_lambda = abs(lambda_rot - lambda_base) / abs(lambda_base)
    assert rel_error_lambda < 1e-08, f'Critical load factor not invariant: lambda_base={lambda_base}, lambda_rot={lambda_rot}, rel_error={rel_error_lambda}'
    n_dofs = 6 * n_nodes
    T = np.zeros((n_dofs, n_dofs))
    for node in range(n_nodes):
        idx = 6 * node
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    mode_base_transformed = T @ mode_base
    norm_rot = np.linalg.norm(mode_rot)
    norm_transformed = np.linalg.norm(mode_base_transformed)
    if norm_rot > 1e-12 and norm_transformed > 1e-12:
        mode_rot_norm = mode_rot / norm_rot
        mode_transformed_norm = mode_base_transformed / norm_transformed
        dot_product = np.dot(mode_rot_norm, mode_transformed_norm)
        assert abs(abs(dot_product) - 1.0) < 1e-06, f'Modes not aligned after transformation: |dot_product|={abs(dot_product)}'

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    E = 210000.0
    nu = 0.3
    L = 15.0
    r = 0.8
    A = np.pi * r ** 2
    I = np.pi * r ** 4 / 4.0
    J = np.pi * r ** 4 / 2.0
    P_euler = np.pi ** 2 * E * I / (4.0 * L ** 2)
    mesh_sizes = [2, 4, 8, 16, 32]
    errors = []
    for n_elements in mesh_sizes:
        n_nodes = n_elements + 1
        node_coords = np.zeros((n_nodes, 3))
        for i in range(n_nodes):
            node_coords[i, 2] = i * L / n_elements
        elements = []
        for i in range(n_elements):
            elem = {'node_i': i, 'node_j': i + 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0.0, 1.0, 0.0])}
            elements.append(elem)
        boundary_conditions = {0: [1, 1, 1, 1, 1, 1]}
        P_ref = 1.0
        nodal_loads = {n_nodes - 1: [0.0, 0.0, -P_ref, 0.0, 0.0, 0.0]}
        (lambda_cr, mode) = fcn(node_coords, elements, boundary_conditions, nodal_loads)
        P_cr_numerical = lambda_cr * P_ref
        rel_error = abs(P_cr_numerical - P_euler) / P_euler
        errors.append(rel_error)
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1], f'Error did not decrease with mesh refinement: n_elements={mesh_sizes[i - 1]} error={errors[i - 1]}, n_elements={mesh_sizes[i]} error={errors[i]}'
    assert errors[-1] < 1e-06, f'Finest mesh (n_elements={mesh_sizes[-1]}) did not achieve sufficient accuracy: rel_error={errors[-1]}'
    for i in range(1, len(errors) - 1):
        if errors[i + 1] > 1e-14:
            ratio = errors[i] / errors[i + 1]
            assert ratio > 3.0, f'Convergence rate slower than expected between n_elements={mesh_sizes[i]} and {mesh_sizes[i + 1]}: ratio={ratio}'