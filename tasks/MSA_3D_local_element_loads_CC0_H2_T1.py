import numpy as np
import pytest
from typing import Optional


def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J):
    """
    Return the 12x12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam element.

    The beam is assumed to be aligned with the local x-axis. The stiffness matrix
    relates local nodal displacements and rotations to forces and moments using the equation:

        [force_vector] = [stiffness_matrix] @ [displacement_vector]

    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Where:
        - u, v, w: displacements along local x, y, z
        - θx, θy, θz: rotations about local x, y, z
        - Subscripts 1 and 2 refer to node i and node j of the element

    Parameters:
        E (float): Young's modulus
        nu (float): Poisson's ratio (used for torsion only)
        A (float): Cross-sectional area
        L (float): Length of the beam element
        Iy (float): Second moment of area about the local y-axis
        Iz (float): Second moment of area about the local z-axis
        J (float): Torsional constant

    Returns:
        np.ndarray: A 12x12 symmetric stiffness matrix representing axial, torsional,
                    and bending stiffness in local coordinates.
    """
    k = np.zeros((12, 12))
    EA_L = E * A / L
    GJ_L = E * J / (2.0 * (1.0 + nu) * L)
    EIz_L = E * Iz
    EIy_L = E * Iy
    # axial
    k[0, 0] = k[6, 6] = EA_L
    k[0, 6] = k[6, 0] = -EA_L
    # torsion
    k[3, 3] = k[9, 9] = GJ_L
    k[3, 9] = k[9, 3] = -GJ_L
    # bending about z (local y‑displacements & rotations about z)
    k[1, 1] = k[7, 7] = 12.0 * EIz_L / L**3
    k[1, 7] = k[7, 1] = -12.0 * EIz_L / L**3
    k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = 6.0 * EIz_L / L**2
    k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -6.0 * EIz_L / L**2
    k[5, 5] = k[11, 11] = 4.0 * EIz_L / L
    k[5, 11] = k[11, 5] = 2.0 * EIz_L / L
    # bending about y (local z‑displacements & rotations about y)
    k[2, 2] = k[8, 8] = 12.0 * EIy_L / L**3
    k[2, 8] = k[8, 2] = -12.0 * EIy_L / L**3
    k[2, 4] = k[4, 2] = k[2, 10] = k[10, 2] = -6.0 * EIy_L / L**2
    k[4, 8] = k[8, 4] = k[8, 10] = k[10, 8] = 6.0 * EIy_L / L**2
    k[4, 4] = k[10, 10] = 4.0 * EIy_L / L
    k[4, 10] = k[10, 4] = 2.0 * EIy_L / L
    return k


def beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, ref_vec: Optional[np.ndarray]):
    """
    Compute the 12x12 transformation matrix Gamma for a 3D beam element.

    This transformation relates the element's local coordinate system to the global system:
        K_global = Gamma.T @ K_local @ Gamma
    where K_global is the global stiffness matrix and K_local is the local stiffness matrix.

    Parameters:
        x1, y1, z1 (float): Coordinates of the beam's start node in global space.
        x2, y2, z2 (float): Coordinates of the beam's end node in global space.
        reference_vector (np.ndarray of shape (3,), optional): A unit vector in global coordinates used to define
            the orientation of the local y-axis. The local y-axis is computed as the cross product
            of the reference vector and the local x-axis (beam axis). The local z-axis is then
            computed as the cross product of the local x-axis and the local y-axes.

            If not provided:
            - If the beam is aligned with the global z-axis, the global y-axis is used.
            - Otherwise, the global z-axis is used.

    Returns:
        Gamma (np.ndarray): A 12x12 local-to-global transformation matrix used to transform
            stiffness matrices, displacements, and forces. It is composed of four repeated
            3x3 direction cosine submatrices along the diagonal.

    Raises:
        ValueError: If `reference_vector` is not a unit vector.
        ValueError: If `reference_vector` is parallel to the beam axis.
        ValueError: If the `reference_vector` doesn't have shape (3,).
        ValueError: If the beam has zero length (start and end node
        s coincide).

    Notes:
        All vectors must be specified in a right-handed global Cartesian coordinate system.
    """
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    L = np.sqrt(dx*dx + dy*dy + dz*dz)
    if np.isclose(L, 0.0):
        raise ValueError("Beam length is zero.")
    ex = np.array([dx, dy, dz]) / L
    if ref_vec is None:
        ref_vec = np.array([0.0, 0.0, 1.0]) if not (np.isclose(ex[0], 0) and np.isclose(ex[1], 0)) \
                    else np.array([0.0, 1.0, 0.0])
    else:
        ref_vec = np.asarray(ref_vec, dtype=float)
        if ref_vec.shape != (3,):
            raise ValueError("local_z/reference_vector must be length‑3.")
        if not np.isclose(np.linalg.norm(ref_vec), 1.0):
            raise ValueError("reference_vector must be unit length.")
        if np.isclose(np.linalg.norm(np.cross(ref_vec, ex)), 0.0):
            raise ValueError("reference_vector parallel to beam axis.")

    ey = np.cross(ref_vec, ex)
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)

    gamma = np.vstack((ex, ey, ez))  # 3×3
    return np.kron(np.eye(4), gamma)  # 12×12


def MSA_3D_local_element_loads_CC0_H2_T1(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    """
    Compute the local internal nodal force/moment vector (end loading) for a 3D Euler-Bernoulli beam element.

    This function transforms the element's global displacement vector into local coordinates,
    applies the local stiffness matrix, and returns the corresponding internal end forces
    in the local coordinate system.

    Parameters
    ----------
    ele_info : dict
        Dictionary containing the element's material and geometric properties:
            'E' : float
                Young's modulus (Pa).
            'nu' : float
                Poisson's ratio (unitless).
            'A' : float
                Cross-sectional area (m²).
            'I_y', 'I_z' : float
                Second moments of area about the local y- and z-axes (m⁴).
            'J' : float
                Torsional constant (m⁴).
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.

    xi, yi, zi : float
        Global coordinates of the element's start node.

    xj, yj, zj : float
        Global coordinates of the element's end node.

    u_dofs_global : array-like of shape (12,)
        Element displacement vector in global coordinates:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2],
        where:
            - u, v, w are translations along x, y, z
            - θx, θy, θz are rotations about x, y, z
            - Subscripts 1 and 2 refer to the start (i) and end (j) nodes.

    Returns
    -------
    load_dofs_local : ndarray of shape (12,)
        Internal element end forces in local coordinates, ordered consistently with `u_dofs_global`.
        Positive forces/moments follow the local right-handed coordinate system conventions.

    Raises
    ------
    ValueError
        If the beam length is zero or if `local_z` is invalid.

    Notes
    -----
    - This computation assumes an Euler-Bernoulli beam (no shear deformation).
    - The returned forces and moments are internal: they represent the element's elastic response
      to the provided displacement state, not externally applied loads.

    Support Functions Used
    ----------------------
    - `beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, local_z)`
        Computes the 12x12 transformation matrix (Gamma) relating local and global coordinate systems
        for a 3D beam element. Ensures orthonormal local axes and validates the reference vector.

    - `local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J)`
        Returns the 12x12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam aligned with the
        local x-axis, capturing axial, bending, and torsional stiffness.
    """
    v = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
    L = np.linalg.norm(v)

    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele_info.get('local_z'))
    k_e_local = local_elastic_stiffness_matrix_3D_beam(
        ele_info['E'], ele_info['nu'], ele_info['A'], L,
        ele_info['I_y'], ele_info['I_z'], ele_info['J']
    )
    u_dofs_local = Gamma @ u_dofs_global
    load_dofs_local = k_e_local @ u_dofs_local
    return load_dofs_local


def compute_local_element_loads_beam_3D_all_ones(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    """
    Dummy version that ignores inputs and just returns all ones.
    This should break:
      - Rigid-body motion test (nonzero loads appear even for rigid translations).
      - Unit response tests (magnitudes/signs are wrong).
      - Linearity/superposition (nonlinear constant output).
      - Coordinate invariance (output doesn't transform with rotation).
    """
    return np.ones(12, dtype=float)


def compute_local_element_loads_beam_3D_return_scaled_disp(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    """
    Dummy: internal loads are just 1000x the displacements.
    This should break:
      - Rigid-body motion test (nonzero loads appear even for rigid translations).
      - Unit response tests (magnitudes/signs are wrong).
      - Coordinate invariance (output doesn't transform with rotation).
    """
    return 1000.0 * np.asarray(u_dofs_global, dtype=float)


def compute_local_element_loads_beam_3D_return_nonlinear_scaled_disp(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
    """Dummy:
    nonlinear mapping, squares the displacements elementwise.
    This should break:
      - Rigid-body motion test (nonzero loads appear even for rigid translations).
      - Unit response tests (magnitudes/signs are wrong).
      - Linearity/superposition (nonlinear constant output).
      - Coordinate invariance (output doesn't transform with rotation).

    """
    u = np.asarray(u_dofs_global, dtype=float)
    s = float(u @ u)              # ||u||^2
    return s * u                  # nonlinear; f(ua+ub) ≠ f(ua)+f(ub)


def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    # Element + properties
    E, nu = 210e9, 0.30
    A, Iy, Iz, J = 3.0e-4, 8.0e-6, 5.0e-6, 2.0e-5
    L = 2.0
    xi, yi, zi = 0.0, 0.0, 0.0
    xj, yj, zj = L,   0.0, 0.0
    ele_info = {"E": E, "nu": nu, "A": A, "I_y": Iy, "I_z": Iz, "J": J, "local_z": np.array([0.0, 0.0, 1.0])}

    # Same translation at both nodes; zero rotations
    t = np.array([0.012, -0.004, 0.007])
    u = np.zeros(12)
    u[0:3] = t
    u[6:9] = t

    load_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)
    assert load_local.shape == (12,)
    assert np.allclose(load_local, 0.0, rtol=1e-9, atol=1e-10)


def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    # Geometry & properties (self-contained)
    E, nu = 210e9, 0.30
    A, Iy, Iz, J = 3.0e-4, 8.0e-6, 5.0e-6, 2.0e-5
    L = 2.0
    xi, yi, zi = 0.0, 0.0, 0.0
    xj, yj, zj = L,   0.0, 0.0
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)

    ele_info = {"E": E, "nu": nu, "A": A, "I_y": Iy, "I_z": Iz, "J": J, "local_z": local_z}

    RTOL, ATOL = 1e-9, 1e-10

    # -------------------------------------------------
    # (1) Axial "unit" extension: u2x - u1x = 1e-4
    # -------------------------------------------------
    u_axial = np.zeros(12, dtype=float)
    u_axial[0] = 0.0     # u1x
    u_axial[6] = 1e-4    # u2x

    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    Fx_i, Fx_j = load_axial[0], load_axial[6]

    expected_axial = E * A / L * (u_axial[6] - u_axial[0])
    # Action–reaction and correct magnitude (node-i sign depends on convention)
    assert np.isclose(Fx_i, -Fx_j, rtol=RTOL, atol=ATOL)
    assert np.isclose(abs(Fx_i), abs(expected_axial), rtol=1e-7, atol=1e-8)
    # Other components ~ 0
    mask_other = np.ones(12, dtype=bool); mask_other[[0, 6]] = False
    assert np.allclose(load_axial[mask_other], 0.0, rtol=1e-6, atol=1e-8)

    # -------------------------------------------------------------------
    # (2) "Unit shear" via transverse y-translation: v2 - v1 = 1e-4, θz=0
    #     Expected (EB beam, bending about z):
    #       V_i =  (12*E*Iz/L^3) * (v1 - v2) + ... (rot terms=0 here)
    #       M_i =  ( 6*E*Iz/L^2) * (v1 - v2)
    #       V_j = -(V_i)
    #       M_j =  ( 6*E*Iz/L^2) * (v1 - v2)   (same sign as M_i for this case)
    # -------------------------------------------------------------------
    u_shear = np.zeros(12, dtype=float)
    v1, v2 = 0.0, 1e-4
    u_shear[1] = v1     # v1 (node i, y-translation)
    u_shear[7] = v2     # v2 (node j, y-translation)
    # θz1 = θz2 = 0 already

    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)

    Vi = load_shear[1]     # Fy_i
    Vj = load_shear[7]     # Fy_j
    Mi = load_shear[5]     # Mz_i
    Mj = load_shear[11]    # Mz_j

    expected_Vi = (12.0 * E * Iz / (L**3)) * (v1 - v2)
    expected_Vj = -expected_Vi
    expected_Mi = (6.0 * E * Iz / (L**2)) * (v1 - v2)
    expected_Mj = expected_Mi

    assert np.isclose(Vi, expected_Vi, rtol=1e-6, atol=1e-8)
    assert np.isclose(Vj, expected_Vj, rtol=1e-6, atol=1e-8)
    assert np.isclose(Mi, expected_Mi, rtol=1e-6, atol=1e-8)
    assert np.isclose(Mj, expected_Mj, rtol=1e-6, atol=1e-8)

    # Other components small
    mask_other2 = np.ones(12, dtype=bool); mask_other2[[1, 7, 5, 11]] = False
    assert np.allclose(load_shear[mask_other2], 0.0, rtol=1e-6, atol=1e-8)

    # -------------------------------------------------
    # (3) Unit torsional rotation: θx2 - θx1 = 1e-4
    #     Expected torsion: Mx = (GJ/L) * Δθx, G=E/(2(1+ν))
    # -------------------------------------------------
    u_torsion = np.zeros(12, dtype=float)
    u_torsion[3] = 0.0     # θx1
    u_torsion[9] = 1e-4    # θx2

    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    Mx_i, Mx_j = load_torsion[3], load_torsion[9]

    G = E / (2.0 * (1.0 + nu))
    expected_M = G * J / L * (u_torsion[9] - u_torsion[3])

    assert np.isclose(Mx_i, -Mx_j, rtol=RTOL, atol=ATOL)
    assert np.isclose(abs(Mx_i), abs(expected_M), rtol=1e-7, atol=1e-8)

    # Other components ~ 0
    mask_other3 = np.ones(12, dtype=bool); mask_other3[[3, 9]] = False
    assert np.allclose(load_torsion[mask_other3], 0.0, rtol=1e-6, atol=1e-8)


def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds.
    """
    # Element + properties
    E, nu = 210e9, 0.30
    A, Iy, Iz, J = 3.0e-4, 8.0e-6, 5.0e-6, 2.0e-5
    L = 2.0
    xi, yi, zi = 0.0, 0.0, 0.0
    xj, yj, zj = L,   0.0, 0.0
    ele_info = {"E": E, "nu": nu, "A": A, "I_y": Iy, "I_z": Iz, "J": J, "local_z": np.array([0.0, 0.0, 1.0])}

    # Two independent displacement patterns
    ua = np.zeros(12); ua[1] = 2e-4; ua[10] = -3e-4   # v1, theta_y2
    ub = np.zeros(12); ub[2] = 1e-4; ub[3]  = 5e-4; ub[9] = 4e-4  # w1, theta_x1, theta_x2
    u_sum = ua + ub

    la = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    lb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    lsum = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_sum)

    assert np.allclose(lsum, la + lb, rtol=1e-9, atol=1e-10)


def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    # Element + properties
    E, nu = 210e9, 0.30
    A, Iy, Iz, J = 3.0e-4, 8.0e-6, 5.0e-6, 2.0e-5
    L = 2.0
    xi, yi, zi = 0.0, 0.0, 0.0
    xj, yj, zj = L,   0.0, 0.0
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)

    ele_info = {"E": E, "nu": nu, "A": A, "I_y": Iy, "I_z": Iz, "J": J, "local_z": local_z}

    # Mixed deformation in GLOBAL components
    u = np.zeros(12, dtype=float)
    u[0] = 1.0e-4   # u1x
    u[7] = -2.5e-4   # v2y
    u[2] = 1.2e-4   # w1z
    u[3] = 4.0e-4   # theta_x1
    u[9] = -1.5e-4   # theta_x2

    # Local internal loads before rotation
    load_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)

    # Rigid global rotation about z by 30°
    theta = np.deg2rad(30.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ], dtype=float)

    # Rotate coordinates
    pi_rot = R @ np.array([xi, yi, zi], dtype=float)
    pj_rot = R @ np.array([xj, yj, zj], dtype=float)
    xi2, yi2, zi2 = pi_rot
    xj2, yj2, zj2 = pj_rot

    # Rotate displacements (translations & rotations)
    def rotate_u12(u12, Rmat):
        out = u12.copy()
        out[0:3] = Rmat @ out[0:3]   # node i translations
        out[3:6] = Rmat @ out[3:6]   # node i rotations
        out[6:9] = Rmat @ out[6:9]   # node j translations
        out[9:12] = Rmat @ out[9:12]  # node j rotations
        return out

    u_rot = rotate_u12(u, R)

    # Rotate local_z too (local frame follows the body)
    ele_info_rot = dict(ele_info)
    ele_info_rot["local_z"] = (R @ local_z).astype(float)

    # Local internal loads after rotation
    load_local_rot = fcn(ele_info_rot, xi2, yi2, zi2, xj2, yj2, zj2, u_rot)

    # Components must be identical in their respective local frames
    assert load_local.shape == (12,)
    assert load_local_rot.shape == (12,)
    assert np.allclose(load_local_rot, load_local, rtol=1e-8, atol=1e-9)


def task_info():
    task_id = "MSA_3D_local_element_loads_CC0_H2_T1"
    task_short_description = "compute the local element loads for a 3D beam element"
    created_date = "2025-09-12"
    created_by = "elejeune11"
    main_fcn = MSA_3D_local_element_loads_CC0_H2_T1
    required_imports = ["import numpy as np", "from typing import Callable, Optional", "import pytest"]
    fcn_dependencies = [local_elastic_stiffness_matrix_3D_beam, beam_transformation_matrix_3D]
    reference_verification_inputs = [
        # 1) +x axis, pure axial extension
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 0.0, 1.0]),
            },
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1e-4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 2) +x axis, torsion
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 0.0, 1.0]),
            },
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1e-4,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 3) +x axis, bending about z (transverse y)
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 0.0, 1.0]),
            },
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1e-4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 4) +x axis, rigid translation (expect near-zero loads)
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 0.0, 1.0]),
            },
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            np.array(
                [
                    1.2e-3,
                    -0.4e-3,
                    0.7e-3,
                    0.0,
                    0.0,
                    0.0,
                    1.2e-3,
                    -0.4e-3,
                    0.7e-3,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 5) 45° in xy plane, mixed DOFs
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 0.0, 1.0]),
            },
            0.0,
            0.0,
            0.0,
            2.0 / np.sqrt(2.0),
            2.0 / np.sqrt(2.0),
            0.0,
            np.array(
                [
                    1.0e-4,
                    0.0,
                    0.8e-4,
                    3.0e-4,
                    0.0,
                    0.0,
                    -2.5e-4,
                    1.0e-4,
                    0.0,
                    -1.5e-4,
                    2.0e-4,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 6) +z axis, axial extension (use y-up reference for vertical)
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 1.0, 0.0]),
            },
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1e-4,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 7) Skew in x–z plane (30° incline), torsion
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 1.0, 0.0]),
            },
            0.0,
            0.0,
            0.0,
            np.sqrt(3.0),
            0.0,
            1.0,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0e-4,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 8) Fully skewed diagonal (x=y=z), bending mix
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 0.0, 1.0]),
            },
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            np.array(
                [
                    0.0,
                    2.0e-4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0e-4,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 9) Negative x, axial compression
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 0.0, 1.0]),
            },
            1.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0e-4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            ),
        ],
        # 10) Random skew, mixed DOFs
        [
            {
                "E": 210e9,
                "nu": 0.30,
                "A": 3.0e-4,
                "I_y": 8.0e-6,
                "I_z": 5.0e-6,
                "J": 2.0e-5,
                "local_z": np.array([0.0, 1.0, 0.0]),
            },
            1.0,
            2.0,
            0.0,
            3.0,
            3.0,
            1.0,
            np.array(
                [
                    0.5e-4,
                    -0.2e-4,
                    0.1e-4,
                    0.0,
                    0.3e-4,
                    0.0,
                    0.0,
                    0.4e-4,
                    -0.3e-4,
                    0.2e-4,
                    0.0,
                    0.1e-4,
                ],
                dtype=float,
            ),
        ],
    ]
    test_cases = [
        {"test_code": test_rigid_body_motion_zero_loads,
         "expected_failures": [compute_local_element_loads_beam_3D_all_ones, compute_local_element_loads_beam_3D_return_scaled_disp, compute_local_element_loads_beam_3D_return_nonlinear_scaled_disp]},
        {"test_code": test_unit_responses_axial_shear_torsion,
         "expected_failures": [compute_local_element_loads_beam_3D_all_ones, compute_local_element_loads_beam_3D_return_scaled_disp, compute_local_element_loads_beam_3D_return_nonlinear_scaled_disp]},
        {"test_code": test_superposition_linearity,
         "expected_failures": [compute_local_element_loads_beam_3D_all_ones, compute_local_element_loads_beam_3D_return_nonlinear_scaled_disp]},
        {"test_code": test_coordinate_invariance_global_rotation,
         "expected_failures": [compute_local_element_loads_beam_3D_return_scaled_disp]},
    ]
    return {
        "task_id": task_id,
        "task_short_description": task_short_description,
        "created_date": created_date,
        "created_by": created_by,
        "main_fcn": main_fcn,
        "required_imports": required_imports,
        "fcn_dependencies": fcn_dependencies,
        "reference_verification_inputs": reference_verification_inputs,
        "test_cases": test_cases,
        # "python_version": "version_number",
        # "package_versions": {"numpy": "version_number", },
    }