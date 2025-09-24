@pytest.mark.parametrize('r', [0.5, 0.75, 1.0])
@pytest.mark.parametrize('L', [10.0, 20.0, 40.0])
def test_euler_buckling_cantilever_circular_param_sweep(fcn, r, L):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever analytical value.
    Uses 10 elements. Checks with tight relative tolerance suitable for high-accuracy EB elements.
    """
    E = 210000000000.0
    nu = 0.3
    nel = 10
    load_mag = 1.0
    (node_coords, elements, bcs, loads) = _build_cantilever_frame_inputs_circular(L=L, nel=nel, r=r, E=E, nu=nu, local_z_vec=(1.0, 0.0, 0.0), load_mag=load_mag)
    (lam, mode) = fcn(node_coords, elements, bcs, loads)
    I = math.pi * r ** 4 / 4.0
    Pcr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
    assert np.isclose(lam * load_mag, Pcr_euler, rtol=0.0001, atol=0.0)
    assert mode.shape == (6 * node_coords.shape[0],)

def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).
    The model is solved in its base orientation and again after applying a rigid-body rotation R
    to geometry, element axes, and applied load. The critical load factor must be identical.
    The buckling mode of the rotated model must equal the base mode transformed by R up to scale/sign.
    """
    L = 12.0
    nel = 12
    E = 190000000000.0
    nu = 0.29
    A = 0.02
    Iy = 3e-05
    Iz = 1e-05
    J = Iy + Iz
    (node_coords, elements, bcs, loads) = _build_cantilever_frame_inputs_rect(L=L, nel=nel, A=A, Iy=Iy, Iz=Iz, J=J, E=E, nu=nu, local_z_vec=(1.0, 0.0, 0.0), load_vec=(0.0, 0.0, -1.0))
    (lam_base, mode_base) = fcn(node_coords, elements, bcs, loads)
    n_nodes = node_coords.shape[0]
    assert mode_base.shape == (6 * n_nodes,)
    axis = (1.0, 1.0, 1.0)
    theta = 0.7
    R = _axis_angle_to_R(axis, theta)
    node_coords_rot = (R @ node_coords.T).T
    local_z_base = np.array([1.0, 0.0, 0.0])
    local_z_rot = (R @ local_z_base.reshape(3, 1)).ravel()
    F_base = np.array([0.0, 0.0, -1.0])
    F_rot = R @ F_base
    elements_rot = []
    for e in elements:
        ed = dict(e)
        ed['local_z'] = list(local_z_rot)
        elements_rot.append(ed)
    loads_rot = {n_nodes - 1: [float(F_rot[0]), float(F_rot[1]), float(F_rot[2]), 0.0, 0.0, 0.0]}
    (lam_rot, mode_rot) = fcn(node_coords_rot, elements_rot, bcs, loads_rot)
    assert np.isclose(lam_rot, lam_base, rtol=1e-09, atol=0.0)
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for i in range(n_nodes):
        blk = np.zeros((6, 6), dtype=float)
        blk[:3, :3] = R
        blk[3:, 3:] = R
        T[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)] = blk
    mode_base_rot_pred = T @ mode_base
    denom = float(np.dot(mode_base_rot_pred, mode_base_rot_pred))
    assert denom > 0.0
    s = float(np.dot(mode_rot, mode_base_rot_pred) / denom)
    rel_err = np.linalg.norm(mode_rot - s * mode_base_rot_pred) / np.linalg.norm(mode_rot)
    assert rel_err < 5e-06

def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    Refine the beam discretization and require that the numerical critical load
    approaches the analytical value with decreasing relative error, achieving
    very high accuracy at the finest mesh.
    """
    E = 200000000000.0
    nu = 0.3
    L = 30.0
    r = 0.6
    load_mag = 1.0
    I = math.pi * r ** 4 / 4.0
    Pcr_euler = math.pi ** 2 * E * I / (4.0 * L ** 2)
    mesh_sizes = [2, 4, 8, 16, 32]
    errors = []
    for nel in mesh_sizes:
        (node_coords, elements, bcs, loads) = _build_cantilever_frame_inputs_circular(L=L, nel=nel, r=r, E=E, nu=nu, local_z_vec=(1.0, 0.0, 0.0), load_mag=load_mag)
        (lam, _) = fcn(node_coords, elements, bcs, loads)
        rel_err = abs(lam * load_mag - Pcr_euler) / Pcr_euler
        errors.append(rel_err)
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] * (1.0 + 1e-12)
    assert errors[-1] < 1e-05