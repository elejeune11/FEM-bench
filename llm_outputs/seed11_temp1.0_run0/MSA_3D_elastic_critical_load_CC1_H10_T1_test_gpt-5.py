@pytest.mark.parametrize('r', [0.5, 0.75, 1.0])
@pytest.mark.parametrize('L', [10.0, 20.0, 40.0])
def test_euler_buckling_cantilever_circular_param_sweep(fcn, r, L):
    """
    Cantilever (fixed-free) circular column aligned with +z. Sweep through radii and lengths.
    Compare λ to Euler critical load for a cantilever with 10 elements. Tight tolerance.
    """
    E = 210000000000.0
    nu = 0.3
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = math.pi * r ** 4 / 2.0
    n_elems = 10
    n_nodes = n_elems + 1
    coords = _build_cantilever_geometry(L, n_elems, R=None)
    elements = _build_elements(n_elems, E, nu, A, Iy, Iz, J, local_z_vec=[0.0, 1.0, 0.0])
    bcs = _fixed_base_bc()
    P_ref = 1.0
    loads = _end_axial_compressive_load(n_nodes, axis_vec=[0.0, 0.0, 1.0], P_ref=P_ref)
    (lam, mode) = fcn(coords, elements, bcs, loads)
    Pcr_euler = _euler_cantilever_Pcr(E, I, L)
    assert lam > 0.0
    rel_err = abs(lam - Pcr_euler) / Pcr_euler
    assert rel_err < 5e-05

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance: Solve a rectangular cantilever in base and rotated configurations.
    The critical load factor must be identical, and mode shapes must match under rigid-body rotation.
    """
    L = 12.0
    n_elems = 12
    n_nodes = n_elems + 1
    E = 210000000000.0
    nu = 0.3
    by = 0.2
    bz = 0.4
    A = by * bz
    Iy = by * bz ** 3 / 12.0
    Iz = bz * by ** 3 / 12.0
    J = Iy + Iz
    local_z_base = np.array([0.0, 1.0, 0.0], dtype=float)
    coords_base = _build_cantilever_geometry(L, n_elems, R=None)
    elements_base = _build_elements(n_elems, E, nu, A, Iy, Iz, J, local_z_vec=local_z_base)
    bcs_base = _fixed_base_bc()
    P_ref = 1.0
    loads_base = _end_axial_compressive_load(n_nodes, axis_vec=[0.0, 0.0, 1.0], P_ref=P_ref)
    (lam_base, mode_base) = fcn(coords_base, elements_base, bcs_base, loads_base)
    R = _rotation_matrix_from_euler(0.7, 0.0, -0.4)
    coords_rot = _build_cantilever_geometry(L, n_elems, R=R)
    local_z_rot = (R @ local_z_base).tolist()
    elements_rot = _build_elements(n_elems, E, nu, A, Iy, Iz, J, local_z_vec=local_z_rot)
    bcs_rot = _fixed_base_bc()
    axis_rot = R @ np.array([0.0, 0.0, 1.0])
    loads_rot = _end_axial_compressive_load(n_nodes, axis_vec=axis_rot, P_ref=P_ref)
    (lam_rot, mode_rot) = fcn(coords_rot, elements_rot, bcs_rot, loads_rot)
    assert lam_base > 0.0 and lam_rot > 0.0
    assert abs(lam_rot - lam_base) / lam_base < 1e-08
    T = _block_T_from_R(R, n_nodes)
    mode_pred = T @ mode_base
    denom = np.dot(mode_pred, mode_pred)
    alpha = np.dot(mode_rot, mode_pred) / denom if denom > 0 else 1.0
    rel_mode_err = np.linalg.norm(mode_rot - alpha * mode_pred) / np.linalg.norm(mode_rot)
    assert rel_mode_err < 1e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Verify decreasing relative error with refinement and high accuracy on the finest mesh.
    """
    E = 210000000000.0
    nu = 0.3
    r = 0.5
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    Iy = I
    Iz = I
    J = math.pi * r ** 4 / 2.0
    L = 10.0
    meshes = [2, 4, 8, 16, 32]
    Pcr_euler = _euler_cantilever_Pcr(E, I, L)
    errors = []
    for n_elems in meshes:
        n_nodes = n_elems + 1
        coords = _build_cantilever_geometry(L, n_elems, R=None)
        elements = _build_elements(n_elems, E, nu, A, Iy, Iz, J, local_z_vec=[0.0, 1.0, 0.0])
        bcs = _fixed_base_bc()
        loads = _end_axial_compressive_load(n_nodes, axis_vec=[0.0, 0.0, 1.0], P_ref=1.0)
        (lam, _) = fcn(coords, elements, bcs, loads)
        rel_err = abs(lam - Pcr_euler) / Pcr_euler
        errors.append(rel_err)
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] * (1.0 + 1e-06)
    assert errors[-1] < 1e-06