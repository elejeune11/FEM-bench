import numpy as np
from typing import Optional, Sequence


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
        np.ndarray: A 12×12 symmetric stiffness matrix representing axial, torsional,
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


def local_geometric_stiffness_matrix_3D_beam(
    L: float,
    A: float,
    I_rho: float,
    Fx2: float,
    Mx2: float,
    My1: float,
    Mz1: float,
    My2: float,
    Mz2: float
) -> np.ndarray:
    """
    Return the 12x12 local geometric stiffness matrix with torsion-bending coupling for a 3D Euler-Bernoulli beam element.

    The beam is assumed to be aligned with the local x-axis. The geometric stiffness matrix is used in conjunction with the elastic stiffness matrix for nonlinear structural analysis.

    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Where:
        - u, v, w: displacements along local x, y, z axes
        - θx, θy, θz: rotations about local x, y, z axes
        - Subscripts 1 and 2 refer to nodes i and j of the element

    Parameters:
        L (float): Length of the beam element [length units]
        A (float): Cross-sectional area [length² units]
        I_rho (float): Polar moment of inertia about the x-axis [length⁴ units]
        Fx2 (float): Internal axial force in the element (positive = tension), evaluated at node 2 [force units]
        Mx2 (float): Torsional moment at node 2 about x-axis [forcexlength units]
        My1 (float): Bending moment at node 1 about y-axis [forcexlength units]
        Mz1 (float): Bending moment at node 1 about z-axis [forcexlength units]
        My2 (float): Bending moment at node 2 about y-axis [forcexlength units]
        Mz2 (float): Bending moment at node 2 about z-axis [forcexlength units]

    Returns:
        np.ndarray: A 12x12 symmetric geometric stiffness matrix in local coordinates.
                    Positive axial force (tension) contributes to element stiffness;
                    negative axial force (compression) can lead to instability.

    Notes:
    Effects captured
        * Axial load stiffening/softening (second order “P-Δ” and “P-θ” effects):
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
        * Coupling of torsion (Mx2) with bending about the local y and z axes.
        * Coupling between end bending moments (My, Mz) and both transverse displacements and rotations (“moment-displacement” and “moment-rotation”).
        * Interaction between the two element nodes (no lumping of geometric terms).

    Implementation details
        * Consistent 12 x 12 matrix for an Euler-Bernoulli beam.
        * Cross-section assumed prismatic and doubly symmetric; no shear deformation, non-uniform torsion, or Wagner warping effects are included.
        * Intended for second-order geometric nonlinear analysis, large displacement static equilibrium, and eigenvalue based buckling calculations.
        * Inputs `Fx2`, `Mx2`, `My1`, `Mz1`, `My2`, `Mz2` are normally taken from the element's internal force vector at the start of the load increment/Newton iteration.
        * Valid in a small strain, moderate rotation framework (Δ L large, ε small).
    """
    k_g = np.zeros((12, 12))
    # upper triangle off diagonal terms
    k_g[0, 6] = -Fx2 / L
    k_g[1, 3] = My1 / L
    k_g[1, 4] = Mx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 9] = My2 / L
    k_g[1, 10] = -Mx2 / L
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 3] = Mz1 / L
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 5] = Mx2 / L
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 9] = Mz2 / L
    k_g[2, 10] = -Fx2 / 10.0
    k_g[2, 11] = -Mx2 / L
    k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
    k_g[3, 5] = (2.0 * My1 - My2) / 6.0
    k_g[3, 7] = -My1 / L
    k_g[3, 8] = -Mz1 / L
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[3, 11] = (My1 + My2) / 6.0
    k_g[4, 7] = -Mx2 / L
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[4, 11] = Mx2 / 2.0
    k_g[5, 7] = -Fx2 / 10.0
    k_g[5, 8] = -Mx2 / L
    k_g[5, 9] = (My1 + My2) / 6.0
    k_g[5, 10] = -Mx2 / 2.0
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 9] = -My2 / L
    k_g[7, 10] = Mx2 / L
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 9] = -Mz2 / L
    k_g[8, 10] = Fx2 / 10.0
    k_g[8, 11] = Mx2 / L
    k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
    k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g


def compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
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
    - Use `Gamma.T @ load_dofs_local` to obtain the same forces in global coordinates.

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
        ValueError: If the beam has zero length (start and end nodes coincide).

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


def assemble_global_geometric_stiffness_3D_beam(
    node_coords: np.ndarray,
    elements: Sequence[dict],
    u_global: np.ndarray,
) -> np.ndarray:
    """
    Assemble the global geometric stiffness matrix K_g for a 3D beam/frames model
    under the current global displacement state.

    This routine loops over 2-node beam elements, forms each element's local
    geometric stiffness `k_g_local` from axial force and end moments computed in
    the local frame, maps it to the global frame via the 12x12 transformation
    `Gamma`, and scatters the result into the global matrix.

    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Cartesian coordinates (x, y, z) for each node, indexed by node id.
    elements : sequence of dict
        Each element dict must provide at least:
          - 'node_i', 'node_j' : int
                End-node indices (0-based).
          - 'A' : float
                Cross-sectional area.
          - 'I_rho' : float
                Polar (or appropriate torsional) second moment used by 
                `local_geometric_stiffness_matrix_3D_beam`
          - 'local_z' : Optional[Sequence[float]]
                Optional approximate local z-axis direction (3,) used to
                disambiguate the element's local triad in the transformation.
        Additional fields may be present.
    u_global : (6*n_nodes,) ndarray of float
        Global displacement DOF vector with 6 DOF per node in this order:
        [u_x, u_y, u_z, θ_x, θ_y, θ_z] for node 0, then node 1, etc.

    Returns
    -------
    K : (6*n_nodes, 6*n_nodes) ndarray of float
        The assembled geometric stiffness matrix.

    External Dependencies
    ---------------------
    beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None) -> (12,12) ndarray
        Returns the standard 3D beam transformation matrix that maps local
        element DOFs to global DOFs (d_g = Γ^T d_l). Assumes 2-node, 6 DOF/node.
    compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global) -> (12,) ndarray
        Computes the element end forces/moments in the local frame
        corresponding to the current displacement state. This function assumes the returned local load vector 
        is: [Fxi, Fyi, Fzi, Mxi, Myi, Mzi, Fxj, Fyj, Fzj, Mxj, Myj, Mzj]
        i.e., translational forces then moments at node i, then node j.
    local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2) -> (12,12) ndarray
        Must return the local geometric stiffness using the element length L, section properties, and local end force resultants as shown.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    u_global = np.asarray(u_global, dtype=float)

    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes

    def _node_dofs6(n: int) -> list[int]:
        base = 6 * n
        return [base + i for i in range(6)]

    K = np.zeros((n_dof, n_dof), dtype=float)

    for ele in elements:
        ni = int(ele["node_i"])
        nj = int(ele["node_j"])
        xi, yi, zi = node_coords[ni]
        xj, yj, zj = node_coords[nj]
        dx, dy, dz = (xj - xi), (yj - yi), (zj - zi)
        L = float(np.linalg.norm([dx, dy, dz]))
        A = float(ele["A"])
        I_rho = float(ele["I_rho"])
        local_z = ele.get("local_z")
        Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
        dofs = _node_dofs6(ni) + _node_dofs6(nj)
        u_e_global = u_global[dofs]
        load_local = compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global)
        Fx2 = float(load_local[6])
        Mx2 = float(load_local[9])
        My1 = float(load_local[4])
        Mz1 = float(load_local[5])
        My2 = float(load_local[10])
        Mz2 = float(load_local[11])
        k_g_local = local_geometric_stiffness_matrix_3D_beam(
            L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2
        )
        k_g_global = Gamma.T @ k_g_local @ Gamma
        K[np.ix_(dofs, dofs)] += k_g_global
    return K


def bad_geom_Kg_nonsymmetric(node_coords, elements, u_global):
    """
    Intentionally non-symmetric and non-linear in u.
    """
    import numpy as np
    n_nodes = len(node_coords)
    ndof = 6 * n_nodes
    factor = float(np.sum(u_global) + 1.0)  # > 0; silly global dependence
    return np.triu(np.ones((ndof, ndof))) * factor  # upper-triangular → non-symmetric


def bad_geom_Kg_axis_only(node_coords, elements, u_global):
    """
    Depends only on global x-positions; ignores orientation transforms.
    """
    import numpy as np
    node_coords = np.asarray(node_coords, dtype=float)
    diag_entries = np.repeat(node_coords[:, 0], 6)
    return np.diag(diag_entries)


def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    # Straight 2-element beam along +x
    L_total = 2.0
    n_nodes = 3
    x = np.linspace(0.0, L_total, n_nodes)
    node_coords = np.c_[x, np.zeros_like(x), np.zeros_like(x)]

    # Section and material properties
    E, nu = 210e9, 0.3
    A = 0.01
    Iy = Iz = 1.0e-6
    J = 2.0e-6
    I_rho = Iy + Iz
    Le = L_total / (n_nodes - 1)

    # Two elements defined by node_i/node_j
    elements = [
        dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz,
             J=J, I_rho=I_rho, local_z=np.array([0.0, 0.0, 1.0])),
        dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz,
             J=J, I_rho=I_rho, local_z=np.array([0.0, 0.0, 1.0]))
    ]

    ndof = 6 * n_nodes

    # 1) Zero displacement → zero matrix
    u0 = np.zeros(ndof)
    Kg0 = fcn(node_coords, elements, u0)
    assert np.allclose(Kg0, 0.0)

    # Nontrivial displacement: axial ramp + small lateral bump
    u = np.zeros(ndof)
    axial_strain = 1e-6
    for n in range(n_nodes):
        u[6*n + 0] = axial_strain * (n * Le)
    u[6*1 + 1] = 1e-5

    Kg = fcn(node_coords, elements, u)

    # 2) Symmetry
    assert np.allclose(Kg, Kg.T, atol=1e-10)

    # 3) Linearity: K_g(αu) = α K_g(u)
    alpha = 1.7
    Kg_scale = fcn(node_coords, elements, alpha * u)
    assert np.allclose(Kg_scale, alpha * Kg, rtol=1e-10, atol=1e-12)

    # 4) Superposition: K_g(u1+u2) = K_g(u1) + K_g(u2)
    u1 = np.zeros(ndof)
    u2 = np.zeros(ndof)
    for n in range(n_nodes):
        u1[6*n + 0] = 0.5e-6 * (n * Le)
    u2[6*1 + 1] = 2e-5
    u2[6*1 + 5] = 1e-5
    Kg1 = fcn(node_coords, elements, u1)
    Kg2 = fcn(node_coords, elements, u2)
    Kg12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(Kg12, Kg1 + Kg2, rtol=1e-10, atol=1e-12)

    # 5) Order independence
    Kg_rev = fcn(node_coords, list(reversed(elements)), u)
    assert np.allclose(Kg, Kg_rev, rtol=1e-12, atol=1e-14)


def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.

    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    # --- Base configuration: straight 2-element beam aligned with +x ---
    L_total = 2.0
    n_nodes = 3
    x = np.linspace(0.0, L_total, n_nodes)
    node_coords = np.c_[x, np.zeros_like(x), np.zeros_like(x)]

    # Section and material properties
    E, nu = 210e9, 0.3
    A = 0.01
    Iy = Iz = 1.0e-6
    J = 2.0e-6
    I_rho = Iy + Iz
    Le = L_total / (n_nodes - 1)

    elements = [
        dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz,
             J=J, I_rho=I_rho, local_z=np.array([0.0, 0.0, 1.0])),
        dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz,
             J=J, I_rho=I_rho, local_z=np.array([0.0, 0.0, 1.0])),
    ]

    ndof = 6 * n_nodes

    # Apply a nontrivial displacement: axial strain, lateral offset, and twist
    u = np.zeros(ndof)
    axial_strain = 2e-6
    for n in range(n_nodes):
        u[6*n + 0] = axial_strain * (n * Le)  # axial stretch
    u[6*1 + 1] = 5e-6                          # lateral displacement
    u[6*1 + 5] = 3e-6                          # small rotation

    Kg = fcn(node_coords, elements, u)

    # --- Define a rigid rotation about z (30°) ---
    theta = np.deg2rad(30.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])

    # Rotate node coordinates
    node_coords_rot = (R @ node_coords.T).T

    # Rotate each element's local z-axis consistently
    elements_rot = []
    for e in elements:
        e_rot = dict(e)
        e_rot["local_z"] = R @ e["local_z"]
        elements_rot.append(e_rot)

    # Rotate displacement vector: per node [u; r] → [R u; R r]
    u_rot = np.zeros_like(u)
    for n in range(n_nodes):
        base = 6*n
        u_rot[base:base+3] = R @ u[base:base+3]     # translations
        u_rot[base+3:base+6] = R @ u[base+3:base+6]   # rotations

    Kg_rot = fcn(node_coords_rot, elements_rot, u_rot)

    # --- Build global DOF transformation T = blockdiag(diag(R,R), …) ---
    T = np.zeros((ndof, ndof))
    for n in range(n_nodes):
        base = 6*n
        T[base:base+3, base:base+3] = R  # translations
        T[base+3:base+6, base+3:base+6] = R  # rotations

    Kg_rot_expected = T @ Kg @ T.T

    # --- Assertions ---
    # Symmetry of K_g should be preserved
    assert np.allclose(Kg, Kg.T, atol=1e-10)
    assert np.allclose(Kg_rot, Kg_rot.T, atol=1e-10)

    # Frame objectivity: rotated system matches congruence transform
    assert np.allclose(Kg_rot, Kg_rot_expected, rtol=1e-9, atol=1e-11), (
        "Geometric stiffness did not transform correctly under global rotation."
    )


def task_info():
    task_id = "assemble_global_geometric_stiffness_3D_beam"
    task_short_description = "assembles a global geometric stiffness matrix for a 3D frame of beam elements"
    created_date = "2025-09-16"
    created_by = "elejeune11"
    main_fcn = assemble_global_geometric_stiffness_3D_beam
    required_imports = ["import numpy as np", "from typing import Optional, Sequence", "import pytest"]
    fcn_dependencies = [local_elastic_stiffness_matrix_3D_beam, local_geometric_stiffness_matrix_3D_beam, compute_local_element_loads_beam_3D, beam_transformation_matrix_3D]
    reference_verification_inputs = [
        # 1) Straight +x chain, axial ramp (3 nodes, 2 elements)
        [
            np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0]]),
            [
                dict(node_i=0, node_j=1, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
                dict(node_i=1, node_j=2, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
            ],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0,
                      2.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
        # 2) Straight +x, with lateral displacement and twist at middle node
        [
            np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [2.0, 0.0, 0.0]]),
            [
                dict(node_i=0, node_j=1, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
                dict(node_i=1, node_j=2, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
            ],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      1.0e-6, 2.0e-6, 0.0, 0.0, 0.0, 1.0e-6,
                      2.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
        # 3) Geometry rotated 45° about z-axis
        [
            np.array([[0.0, 0.0, 0.0],
                      [np.sqrt(2)/2, np.sqrt(2)/2, 0.0],
                      [np.sqrt(2),   np.sqrt(2),   0.0]]),
            [
                dict(node_i=0, node_j=1, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
                dict(node_i=1, node_j=2, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
            ],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      1.0e-6, 1.0e-6, 0.0, 0.0, 0.0, 0.0,
                      2.0e-6, 2.0e-6, 0.0, 0.0, 0.0, 0.0]),
        ],
        # 4) Middle node raised in z, torsional rotation
        [
            np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.5],
                      [2.0, 0.0, 0.0]]),
            [
                dict(node_i=0, node_j=1, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
                dict(node_i=1, node_j=2, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
            ],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      1.0e-6, 0.0, 0.0, 2.0e-6, 0.0, 0.0,
                      2.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ],
        # 5) Diagonal chain in x–y plane, with mixed DOFs
        [
            np.array([[0.0, 0.0, 0.0],
                      [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0],
                      [2.0/np.sqrt(2), 2.0/np.sqrt(2), 0.0]]),
            [
                dict(node_i=0, node_j=1, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
                dict(node_i=1, node_j=2, E=210e9, nu=0.30, A=3.0e-4,
                     I_y=8.0e-6, I_z=5.0e-6, J=2.0e-5, I_rho=1.3e-5,
                     local_z=np.array([0.0, 0.0, 1.0])),
            ],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      5.0e-7, 5.0e-7, 0.0, 0.0, -1.0e-6, 0.0,
                      1.0e-6, 1.0e-6, 0.0, 0.0, 0.0, 0.0]),
        ],
    ]
    test_cases = [{"test_code": test_multi_element_core_correctness_assembly, "expected_failures": [bad_geom_Kg_nonsymmetric, bad_geom_Kg_axis_only]},
                  {"test_code": test_frame_objectivity_under_global_rotation, "expected_failures": [bad_geom_Kg_nonsymmetric, bad_geom_Kg_axis_only]}]
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
