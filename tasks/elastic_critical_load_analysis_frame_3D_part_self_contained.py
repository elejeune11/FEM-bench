import numpy as np
import scipy
from typing import Optional, Sequence


def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J):
    """
    Return the 12x12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam element.
    DOF order (local): [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
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
    # bending about z (local y)
    k[1, 1] = k[7, 7] = 12.0 * EIz_L / L**3
    k[1, 7] = k[7, 1] = -12.0 * EIz_L / L**3
    k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = 6.0 * EIz_L / L**2
    k[5, 7] = k[7, 5] = k[7, 11] = k[11, 7] = -6.0 * EIz_L / L**2
    k[5, 5] = k[11, 11] = 4.0 * EIz_L / L
    k[5, 11] = k[11, 5] = 2.0 * EIz_L / L
    # bending about y (local z)
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
    Return the 12x12 local geometric stiffness matrix with torsion-bending coupling
    for a 3D Euler-Bernoulli beam element.
    DOF order (local): [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
    Positive axial force (tension) stiffens; compression softens.
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
    # symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # diagonal terms
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


def elastic_critical_load_analysis_frame_3D_part_self_contained(
    node_coords: np.ndarray,
    elements: Sequence[dict],
    boundary_conditions: dict[int, Sequence[int | bool]],
    nodal_loads: dict[int, Sequence[float]],
):
    """
    Perform linear (eigenvalue) buckling analysis for a 3D frame and return the
    elastic critical load factor and associated global buckling mode shape.

    Overview
    --------
    The routine:
      1) Assembles the global elastic stiffness matrix `K`.
      2) Assembles the global reference load vector `P`.
      3) Solves the linear static problem `K u = P` (with boundary conditions) to
         obtain the displacement state `u` under the reference load.
      4) Assembles the geometric stiffness `K_g` consistent with that state.
      5) Solves the generalized eigenproblem on the free DOFs,
             K φ = -λ K_g φ,
         and returns the smallest positive eigenvalue `λ` as the elastic
         critical load factor and its corresponding global mode shape `φ`
         (constrained DOFs set to zero).

    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Cartesian coordinates (x, y, z) for each node, indexed 0..n_nodes-1.
    elements : Sequence[dict]
        Element definitions consumed by the assembly routines. Each dictionary
        must supply properties for a 2-node 3D Euler-Bernoulli beam aligned with
        its local x-axis. Required keys (minimum):
          Topology
          --------
          - 'node_i' : int
                Start node index (0-based).
          - 'node_j' : int
                End node index (0-based).
          Material
          --------
          - 'E' : float
                Young's modulus (used in axial, bending, and torsion terms).
          - 'nu' : float
                Poisson's ratio (used in torsion only).
          Section (local axes y,z about the beam's local x)
          -----------------------------------------------
          - 'A'  : float
                Cross-sectional area.
          - 'Iy' : float
                Second moment of area about local y.
          - 'Iz' : float
                Second moment of area about local z.
          - 'J'  : float
                Torsional constant (for elastic/torsional stiffness).
          - 'I_rho' : float
                Polar moment about the local x-axis used by the geometric stiffness
                with torsion–bending coupling.
          Orientation
          -----------
          - 'local_z' : Sequence[float] of length 3 **or** None
                Provide a 3-vector giving the direction of the element's local z-axis to 
                disambiguate the local triad used in the 12×12 transformation; if `None`, 
                a default convention is applied.

    boundary_conditions : dict
        Dictionary mapping node index -> boundary condition specification. Each
        node's specification can be provided in either of two forms:
        - Sequence of 6 booleans: [ux, uy, uz, rx, ry, rz], where True means
          the DOF is constrained (fixed).
        - Sequence of integer indices: e.g. [0, 1, 2] means the DOFs ux, uy, uz
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller’s
        responsibility to supply constraints sufficient to eliminate rigid-body
        modes.

    nodal_loads : dict[int, Sequence[float]]
        Mapping from node index → length-6 vector of load components applied at
        that node in the **global** DOF order `[F_x, F_y, F_z, M_x, M_y, M_z]`.
        Used to form `P`.

    Returns
    -------
    elastic_critical_load_factor : float
        The smallest positive eigenvalue `λ` (> 0). If `P` is the reference load
        used to form `K_g`, then the predicted elastic buckling load is
        `P_cr = λ · P`.
    deformed_shape_vector : (6*n_nodes,) ndarray of float
        Global buckling mode vector with constrained DOFs set to zero. No
        normalization is applied (mode scale is arbitrary; only the shape matters).

    Assumptions
    -----------
    - Frame elements possess 6 DOF per node and use global DOF order
      `[u_x, u_y, u_z, θ_x, θ_y, θ_z]`.
    - Small-displacement (eigenvalue) buckling theory: geometric nonlinearity is
      represented via `K_g` assembled at the reference load state, not via a full
      nonlinear equilibrium/path-following analysis.
    - Boundary conditions must remove rigid-body modes from the free set.
    - Units are consistent across coordinates, properties, and loads.

    External Helper Functions (required)
    ------------------------------------
    - `local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J) -> (12,12) ndarray`
        Local elastic stiffness matrix for a 3D Euler-Bernoulli beam aligned with
        the local x-axis.
    - `local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2) -> (12,12) ndarray`
        Local geometric stiffness matrix with torsion-bending coupling.

    Raises
    ------
    ValueError
        Propagated from called routines if:
        - BCs are invalid or leave rigid-body modes in the free set.
        - Reduced matrices are singular/ill-conditioned beyond tolerances.
        - No positive eigenvalue is found or eigenpairs are significantly complex.

    Notes
    -----
    - The scale of `deformed_shape_vector` is arbitrary; normalize for plotting
      if desired (e.g., by max absolute translational DOF).
    - If multiple nearly-equal smallest eigenvalues exist (mode multiplicity),
      the returned mode can depend on numerical details.
    """
    def beam_transformation_matrix_3D(x1, y1, z1, x2, y2, z2, ref_vec: Optional[np.ndarray]):
        """
        12x12 local-to-global transformation matrix Γ assembled from a 3×3 direction cosine matrix.
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
                raise ValueError("local_z/reference_vector must be length-3.")
            if not np.isclose(np.linalg.norm(ref_vec), 1.0):
                raise ValueError("reference_vector must be unit length.")
            if np.isclose(np.linalg.norm(np.cross(ref_vec, ex)), 0.0):
                raise ValueError("reference_vector parallel to beam axis.")
        ey = np.cross(ref_vec, ex)
        ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        gamma = np.vstack((ex, ey, ez))  # 3×3
        return np.kron(np.eye(4), gamma)  # 12×12

    def assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements):
        n_nodes = node_coords.shape[0]
        n_dof = 6 * n_nodes

        def _node_dofs(n):  # 6 global DOFs for node n
            return list(range(6*n, 6*n + 6))

        K = np.zeros((n_dof, n_dof))
        for ele in elements:
            ni, nj = int(ele['node_i']), int(ele['node_j'])
            xi, yi, zi = node_coords[ni]
            xj, yj, zj = node_coords[nj]
            L = np.linalg.norm([xj - xi, yj - yi, zj - zi])
            Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
            k_loc = local_elastic_stiffness_matrix_3D_beam(
                ele['E'], ele['nu'], ele['A'], L, ele['I_y'], ele['I_z'], ele['J']
            )
            k_glb = Gamma.T @ k_loc @ Gamma
            dofs = _node_dofs(ni) + _node_dofs(nj)
            K[np.ix_(dofs, dofs)] += k_glb
        return K

    def assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes):
        n_dof = 6 * n_nodes

        def _node_dofs(n):
            return list(range(6*n, 6*n + 6))

        P = np.zeros(n_dof)
        for n, load in nodal_loads.items():
            P[_node_dofs(n)] += np.asarray(load, dtype=float)
        return P

    def partition_degrees_of_freedom(boundary_conditions, n_nodes):
        n_dof = n_nodes * 6
        fixed = []
        for n in range(n_nodes):
            spec = boundary_conditions.get(n)
            if spec is None:
                continue
            spec = list(spec)
            if len(spec) == 6 and all(isinstance(v, (bool, np.bool_)) for v in spec):
                fixed.extend([6*n + i for i, f in enumerate(spec) if f])
            else:
                fixed.extend([6*n + int(i) for i in spec])
        fixed = np.asarray(sorted(set(fixed)), dtype=int)
        free = np.setdiff1d(np.arange(n_dof, dtype=int), fixed, assume_unique=True)
        return fixed, free

    def linear_solve(P_global, K_global, fixed, free):
        n_dof = K_global.shape[0]
        K_ff = K_global[np.ix_(free, free)]
        K_sf = K_global[np.ix_(fixed, free)] if fixed.size else np.zeros((0, K_ff.shape[0]))
        condition_number = np.linalg.cond(K_ff)
        if condition_number >= 1e16 or not np.isfinite(condition_number):
            raise ValueError(f"Cannot solve system: stiffness matrix is ill-conditioned (cond={condition_number:.2e})")
        u_f = np.linalg.solve(K_ff, P_global[free])
        u = np.zeros(n_dof)
        u[free] = u_f
        nodal_reaction_vector = np.zeros(n_dof)
        if fixed.size:
            nodal_reaction_vector[fixed] = K_sf @ u_f - P_global[fixed]
        return u, nodal_reaction_vector

    def compute_local_element_loads_beam_3D(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global):
        """
        Uses the external KE to map local displacements to local end forces.
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

    def assemble_global_geometric_stiffness_3D_beam(
        node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray
    ) -> np.ndarray:
        node_coords_ = np.asarray(node_coords, dtype=float)
        u_global_ = np.asarray(u_global, dtype=float)
        n_nodes = node_coords_.shape[0]
        n_dof = 6 * n_nodes

        def _node_dofs6(n: int) -> list[int]:
            base = 6 * n
            return [base + i for i in range(6)]

        K = np.zeros((n_dof, n_dof), dtype=float)
        for ele in elements:
            ni = int(ele["node_i"])
            nj = int(ele["node_j"])
            xi, yi, zi = node_coords_[ni]
            xj, yj, zj = node_coords_[nj]
            dx, dy, dz = (xj - xi), (yj - yi), (zj - zi)
            L = float(np.linalg.norm([dx, dy, dz]))
            A = float(ele["A"])
            I_rho = float(ele["I_rho"])
            local_z = ele.get("local_z")
            Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z)
            dofs = _node_dofs6(ni) + _node_dofs6(nj)
            u_e_global = u_global_[dofs]
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

    def eigenvalue_analysis(K_e_global, K_g_global, boundary_conditions, n_nodes):
        dof_per_node = 6
        cond_limit_e = 1e16
        cond_limit_g = 1e16
        c_eps = 1e3 * np.finfo(float).eps  # ~2e-13

        _, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
        free = np.asarray(free, dtype=int)

        K_e_ff = 0.5 * (K_e_global[np.ix_(free, free)] + K_e_global[np.ix_(free, free)].T)
        K_g_ff = 0.5 * (K_g_global[np.ix_(free, free)] + K_g_global[np.ix_(free, free)].T)

        cond_e = np.linalg.cond(K_e_ff)
        if not np.isfinite(cond_e) or cond_e > cond_limit_e:
            raise ValueError(f"Elastic stiffness (free-free) is ill-conditioned (cond={cond_e:.3e}).")
        cond_g = np.linalg.cond(K_g_ff)
        if not np.isfinite(cond_g) or cond_g > cond_limit_g:
            raise ValueError(f"Geometric stiffness (free-free) is ill-conditioned (cond={cond_g:.3e}).")

        eig_vals, eig_vecs = scipy.linalg.eig(K_e_ff, -1.0 * K_g_ff, check_finite=False)
        if eig_vals.size == 0:
            raise ValueError("Eigen-solution returned no eigenvalues.")

        lam_max = float(np.max(np.abs(eig_vals)))
        rel_imag = np.max(np.abs(np.imag(eig_vals))) / max(lam_max, 1.0)

        if rel_imag <= c_eps:
            V = eig_vecs.copy()
            for j in range(V.shape[1]):
                col = V[:, j]
                k = int(np.argmax(np.abs(col)))
                phase = np.conj(col[k]) / (abs(col[k]) + 1e-300)
                V[:, j] = phase * col
            eig_vals = np.real(eig_vals)
            eig_vecs = np.real(V)
        else:
            raise ValueError(
                "Eigen-solution returned significantly complex eigenpairs "
                f"(relative imag={rel_imag:.3e})."
            )

        vals = eig_vals.astype(float, copy=False)
        finite = np.isfinite(vals)
        lam_scale = np.max(np.abs(vals[finite])) if np.any(finite) else 1.0
        pos_cut = max(1e-12, c_eps * lam_scale)

        cand = np.flatnonzero(finite & (vals > pos_cut))
        if cand.size == 0:
            raise ValueError("No positive buckling factors found.")

        ix = cand[np.argmin(vals[cand])]
        eig_value = float(vals[ix])
        eig_vector_free = eig_vecs[:, ix].astype(float, copy=False)

        num_dofs = dof_per_node * n_nodes
        deformed_shape_vector = np.zeros((num_dofs,), dtype=float)
        deformed_shape_vector[free] = eig_vector_free
        return eig_value, deformed_shape_vector

    n_nodes = node_coords.shape[0]
    K = assemble_global_stiffness_matrix_linear_elastic_3D(node_coords, elements)
    P = assemble_global_load_vector_linear_elastic_3D(nodal_loads, n_nodes)
    fixed, free = partition_degrees_of_freedom(boundary_conditions, n_nodes)
    u_global, _ = linear_solve(P, K, fixed, free)
    K_g = assemble_global_geometric_stiffness_3D_beam(node_coords, elements, u_global)
    elastic_critical_load_factor, deformed_shape_vector = eigenvalue_analysis(
        K, K_g, boundary_conditions, n_nodes
    )
    return elastic_critical_load_factor, deformed_shape_vector


def elastic_critical_load_analysis_frame_3D_all_zeros(
    node_coords, elements, boundary_conditions, nodal_loads
):
    """
    Expected failure: returns λ = 0 and a zero mode.
    """
    n_nodes = int(node_coords.shape[0])
    ndof = 6 * n_nodes
    lam = 0.0
    mode = np.zeros(ndof, dtype=float)
    return lam, mode


def elastic_critical_load_analysis_frame_3D_all_ones(
    node_coords, elements, boundary_conditions, nodal_loads
):
    """
    Expected failure: returns λ = 1 and a mode of all ones (no BC embedding).
    """
    n_nodes = int(node_coords.shape[0])
    ndof = 6 * n_nodes
    lam = 1.0
    mode = np.ones(ndof, dtype=float)
    return lam, mode


def test_euler_buckling_cantilever_circular_param_sweep(fcn):
    """
    Cantilever (fixed-free) circular column aligned with +z.
    Sweep through radii r ∈ {0.5, 0.75, 1.0} and lengths L ∈ {10, 20, 40}.
    For each case, run the full pipeline and compare λ·P_ref to the Euler cantilever value analytical solution.
    Use 10 elements, set tolerances to be appropriate for the anticipated discretization error at 10-5.
    """
    E = 1000.0
    nu = 0.3
    num_nodes = 11
    n_elems = num_nodes - 1
    P_ref = 1.0  # unit reference load makes λ ≈ P_cr directly

    lengths = [10.0, 20.0, 40.0]
    radii = [0.5, 0.75, 1.0]

    for L in lengths:
        for r in radii:
            # Geometry along +z
            z = np.linspace(0.0, L, num_nodes)
            node_coords = np.c_[np.zeros_like(z), np.zeros_like(z), z]

            # Section properties for a circle
            A     = np.pi * r**2
            I_y   = np.pi * r**4 / 4.0
            I_z   = np.pi * r**4 / 4.0
            I_rho = np.pi * r**4 / 2.0
            J     = np.pi * r**4 / 2.0

            # Elements along +z with local_z ≠ axis
            elements = [
                dict(
                    node_i=i, node_j=i + 1,
                    E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, I_rho=I_rho,
                    local_z=np.array([1.0, 0.0, 0.0], dtype=float),
                )
                for i in range(n_elems)
            ]

            # Clamp base, free tip
            boundary_conditions = {0: [True, True, True, True, True, True]}

            # Reference load: compressive axial at tip along -z
            nodal_loads = {n: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for n in range(num_nodes)}
            nodal_loads[num_nodes - 1][2] = -P_ref

            # Run analysis
            lam, mode = fcn(
                node_coords=node_coords,
                elements=elements,
                boundary_conditions=boundary_conditions,
                nodal_loads=nodal_loads,
            )

            # Sanity checks
            assert np.isfinite(lam) and lam > 0.0
            assert mode.shape == (6 * num_nodes,)
            assert np.all(np.isfinite(mode))

            # Euler cantilever reference
            I = I_z
            Pcr_analytical = (np.pi**2) * E * I / (4.0 * L**2)
            Pcr_numeric = lam * P_ref
            rel_err = abs(Pcr_numeric - Pcr_analytical) / Pcr_analytical

            assert rel_err < 1e-5, (
                f"Cantilever Euler mismatch for L={L}, r={r}: "
                f"rel_err={rel_err:.3e} "
                f"(Pnum={Pcr_numeric:.6e}, Panal={Pcr_analytical:.6e}, lam={lam:.6g})"
            )


def test_orientation_invariance_cantilever_buckling_rect_section(fcn):
    """
    Orientation invariance test with a rectangular section (Iy ≠ Iz).

    The cantilever model is solved in its original orientation and again after applying
    a rigid-body rotation R to the geometry, element axes, and applied load. The critical
    load factor λ should be identical in both cases.

    The buckling mode from the rotated model should equal the base mode transformed by R:
    - Build T as a block-diagonal matrix with R applied to both the translational
      [ux, uy, uz] and rotational [θx, θy, θz] DOFs at each node.
    - Then mode_rot ≈ T @ mode_base, allowing for arbitrary scale and sign.
    """
    # --- Base cantilever model ---
    E, nu = 1000.0, 0.30
    L = 2.0
    num_nodes = 11
    n_elems = num_nodes - 1

    # Geometry along +z
    z = np.linspace(0.0, L, num_nodes)
    node_coords = np.c_[np.zeros_like(z), np.zeros_like(z), z]

    # Rectangular cross-section: width b along local y, thickness h along local z
    b = 0.08
    h = 0.05
    A  = b * h
    Iy = b * h**3 / 12.0
    Iz = h * b**3 / 12.0
    if b < h:
        b, h = h, b
    J = b * h**3 * (1/3 - 0.21 * (h/b) * (1 - (h**4) / (12 * b**4)))
    I_rho = Iy + Iz

    elements = [
        dict(
            node_i=i, node_j=i + 1,
            E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, I_rho=I_rho,
            local_z=np.array([0.0, 1.0, 0.0], dtype=float),  # benign triad for a +z member
        )
        for i in range(n_elems)
    ]

    # Clamp base (all 6 DOFs), tip free
    boundary_conditions = {0: [True, True, True, True, True, True]}

    # Reference load: unit compression at free tip along -z
    P_ref = 1.0
    nodal_loads = {n: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for n in range(num_nodes)}
    nodal_loads[num_nodes - 1][2] = -P_ref

    # Solve base problem
    lam_base, mode_base = fcn(
        node_coords=node_coords,
        elements=elements,
        boundary_conditions=boundary_conditions,
        nodal_loads=nodal_loads,
    )

    # --- Build a rigid rotation R (35° about a non-collinear axis) ---
    a = np.array([2.0, -1.0, 0.5], dtype=float)
    a /= np.linalg.norm(a)
    x, y, zax = a
    theta = np.deg2rad(35.0)
    c, s, C = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    R = np.array([
        [c + x*x*C,     x*y*C - zax*s, x*zax*C + y*s],
        [y*x*C + zax*s, c + y*y*C,     y*zax*C - x*s],
        [zax*x*C - y*s, zax*y*C + x*s, c + zax*zax*C],
    ], dtype=float)

    # Rotate node coordinates
    node_coords_rot = (R @ node_coords.T).T

    # Rotate each element's local z
    elements_rot = []
    for e in elements:
        e_rot = dict(e)
        e_rot["local_z"] = R @ e["local_z"]
        elements_rot.append(e_rot)

    # Rotate the tip load vector [0,0,-P_ref]
    F_tip_base = np.array([0.0, 0.0, -P_ref], dtype=float)
    F_tip_rot = R @ F_tip_base
    nodal_loads_rot = {n: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for n in range(num_nodes)}
    nodal_loads_rot[num_nodes - 1][0] = F_tip_rot[0]
    nodal_loads_rot[num_nodes - 1][1] = F_tip_rot[1]
    nodal_loads_rot[num_nodes - 1][2] = F_tip_rot[2]

    # Same BCs (clamped base)
    boundary_conditions_rot = {0: [True, True, True, True, True, True]}

    # Solve rotated problem
    lam_rot, mode_rot = fcn(
        node_coords=node_coords_rot,
        elements=elements_rot,
        boundary_conditions=boundary_conditions_rot,
        nodal_loads=nodal_loads_rot,
    )

    # --- Check λ invariance ---
    assert np.isfinite(lam_base) and lam_base > 0
    assert np.isfinite(lam_rot) and lam_rot > 0
    assert np.isclose(lam_rot, lam_base, rtol=1e-9, atol=0.0), (
        f"λ not invariant under rigid rotation: base={lam_base:.6e}, rot={lam_rot:.6e}"
    )

    # --- Check mode transforms: mode_rot ≈ T @ mode_base (up to sign/scale)
    ndof = 6 * num_nodes
    T = np.zeros((ndof, ndof), dtype=float)
    for n in range(num_nodes):
        bidx = 6 * n
        T[bidx:bidx+3,     bidx:bidx+3]   = R  # translations
        T[bidx+3:bidx+6,   bidx+3:bidx+6] = R  # rotations

    mode_expected = T @ mode_base

    # Compare on free DOFs only (node 0 fixed: DOFs 0..5)
    free = np.arange(6, ndof, dtype=int)
    v1, v2 = mode_rot[free], mode_expected[free]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    assert n1 > 0 and n2 > 0, "Degenerate mode encountered."

    # Normalize and account for sign ambiguity
    v1n, v2n = v1 / n1, v2 / n2
    err = min(np.linalg.norm(v1n - v2n), np.linalg.norm(v1n + v2n))
    assert err < 1e-5, f"Rotated mode does not match T @ base mode (err={err:.3e})"


def test_cantilever_euler_buckling_mesh_convergence(fcn):
    """
    Verify mesh convergence for Euler buckling of a fixed–free circular cantilever.
    The test refines the beam discretization and checks that the numerical critical load
    approaches the analytical Euler value with decreasing relative error, and that the
    finest mesh achieves very high accuracy.
    """
    # --- Problem setup (fixed for the sweep) ---
    E, nu = 1000.0, 0.30
    L = 20.0
    r = 1.0
    P_ref = 1.0  # unit reference → λ ≈ P_cr directly

    # Circular section properties
    A     = np.pi * r**2
    I_y   = np.pi * r**4 / 4.0
    I_z   = np.pi * r**4 / 4.0
    I_rho = np.pi * r**4 / 2.0
    J     = np.pi * r**4 / 2.0

    # Analytical Euler buckling load for cantilever about either principal axis
    I = I_z
    Pcr_exact = (np.pi**2) * E * I / (4.0 * L**2)

    # Mesh refinements (elements); increase as needed
    meshes = [10, 20, 30, 40]

    rel_errors = []

    for n_elems in meshes:
        n_nodes = n_elems + 1

        # Geometry: straight along +z
        z = np.linspace(0.0, L, n_nodes)
        node_coords = np.c_[np.zeros_like(z), np.zeros_like(z), z]

        # Elements: local_z chosen orthogonal to axis (+x) for a well-defined triad
        elements = [
            dict(
                node_i=i, node_j=i + 1,
                E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, I_rho=I_rho,
                local_z=np.array([1.0, 0.0, 0.0], dtype=float),
            )
            for i in range(n_elems)
        ]

        # Boundary conditions: clamp base node (all 6 DOFs fixed), tip free
        boundary_conditions = {0: [True, True, True, True, True, True]}

        # Reference load: compressive axial at the free tip (−z)
        # Per-node load vector: [Fx, Fy, Fz, Mx, My, Mz]
        nodal_loads = {n: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for n in range(n_nodes)}
        nodal_loads[n_nodes - 1][2] = -P_ref

        # Solve
        lam, mode = fcn(
            node_coords=node_coords,
            elements=elements,
            boundary_conditions=boundary_conditions,
            nodal_loads=nodal_loads,
        )

        # Sanity checks
        assert np.isfinite(lam) and lam > 0.0
        assert mode.shape == (6 * n_nodes,)
        assert np.all(np.isfinite(mode))

        # Relative error vs Euler
        Pcr_num = lam * P_ref
        rel_err = abs(Pcr_num - Pcr_exact) / Pcr_exact
        rel_errors.append(rel_err)

    # Monotone improvement (allow tiny numerical wiggle)
    for i in range(1, len(rel_errors)):
        assert rel_errors[i] <= rel_errors[i - 1] * 1.02 + 1e-12, (
            f"Error did not decrease with refinement at step {i}: "
            f"{rel_errors[i-1]:.3e} → {rel_errors[i]:.3e} for meshes {meshes[i-1]}→{meshes[i]}"
        )

    # Finest mesh should be quite accurate
    assert rel_errors[-1] < 1e-6, (
        f"Finest mesh error too large: {rel_errors[-1]:.3e} "
        f"(meshes={meshes}, errors={rel_errors})"
    )


def task_info():
    task_id = "elastic_critical_load_analysis_frame_3D_part_self_contained"
    task_short_description = "performs elastic critical load analysis given problem setup and the elastic and geometric stiffnes matricies pre-defined as helper functions"
    created_date = "2025-09-18"
    created_by = "elejeune11"
    main_fcn = elastic_critical_load_analysis_frame_3D_part_self_contained
    required_imports = ["import numpy as np", "import scipy", "from typing import Optional, Sequence", "import pytest"]
    fcn_dependencies = [local_elastic_stiffness_matrix_3D_beam, local_geometric_stiffness_matrix_3D_beam]
    reference_verification_inputs = [
        # 1) Short cantilever (2 nodes), circular, axial tip load (axis-aligned +z)
        [
            np.array([[0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=float),
            [dict(node_i=0, node_j=1, E=1.0e5, nu=0.30,
                A=np.pi*0.20**2,
                I_y=np.pi*0.20**4/4, I_z=np.pi*0.20**4/4,
                J=np.pi*0.20**4/2, I_rho=np.pi*0.20**4/2,
                local_z=np.array([1.0, 0.0, 0.0], dtype=float))],
            {0: [True, True, True, True, True, True]},
            {1: [0.0, 0.0, -10.0, 0.0, 0.0, 0.0]},
        ],

        # 2) Longer cantilever (2 nodes), rectangular, rotated ~30° about y
        [
            np.array([[0.0, 0.0, 0.0],
                    [4.33, 0.0, 2.5]], dtype=float),  # tip rotated in x-z plane
            [dict(node_i=0, node_j=1, E=2.0e5, nu=0.25,
                A=0.08*0.05,
                I_y=0.08*0.05**3/12, I_z=0.05*0.08**3/12,
                J=1.0e-3, I_rho=1.0e-3,
                local_z=np.array([0.0, 1.0, 0.0], dtype=float))],
            {0: [True, True, True, True, True, True]},
            {1: [-25.0, 0.0, -43.3, 0.0, 0.0, 0.0]},  # axial compression along element axis
        ],

        # 3) Two-element cantilever (3 nodes), circular, rotated and with small lateral
        [
            np.array([[0.0, 0.0, 0.0],
                    [0.94, -0.34, 0.94],
                    [1.88, -0.68, 1.88]], dtype=float),  # roughly rotated about x and z
            [
                dict(node_i=0, node_j=1, E=1.5e5, nu=0.30,
                    A=np.pi*0.25**2,
                    I_y=np.pi*0.25**4/4, I_z=np.pi*0.25**4/4,
                    J=np.pi*0.25**4/2, I_rho=np.pi*0.25**4/2,
                    local_z=np.array([1.0, 0.0, 0.0], dtype=float)),
                dict(node_i=1, node_j=2, E=1.5e5, nu=0.30,
                    A=np.pi*0.25**2,
                    I_y=np.pi*0.25**4/4, I_z=np.pi*0.25**4/4,
                    J=np.pi*0.25**4/2, I_rho=np.pi*0.25**4/2,
                    local_z=np.array([1.0, 0.0, 0.0], dtype=float)),
            ],
            {0: [True, True, True, True, True, True]},
            {2: [-14.1, 5.0, -14.1, 0.0, 0.0, 0.0]},  # axial + small lateral
        ],

        # 4) Three-element cantilever (4 nodes), rectangular, rotated ~45° about z, tip moment
        [
            np.array([[0.0, 0.0, 0.0],
                    [1.06, 1.06, 1.5],
                    [2.12, 2.12, 3.0],
                    [3.18, 3.18, 4.5]], dtype=float),
            [
                dict(node_i=0, node_j=1, E=1.0e5, nu=0.30,
                    A=0.10*0.06,
                    I_y=0.10*0.06**3/12, I_z=0.06*0.10**3/12,
                    J=2.0e-3, I_rho=2.0e-3,
                    local_z=np.array([0.0, 1.0, 0.0], dtype=float)),
                dict(node_i=1, node_j=2, E=1.0e5, nu=0.30,
                    A=0.10*0.06,
                    I_y=0.10*0.06**3/12, I_z=0.06*0.10**3/12,
                    J=2.0e-3, I_rho=2.0e-3,
                    local_z=np.array([0.0, 1.0, 0.0], dtype=float)),
                dict(node_i=2, node_j=3, E=1.0e5, nu=0.30,
                    A=0.10*0.06,
                    I_y=0.10*0.06**3/12, I_z=0.06*0.10**3/12,
                    J=2.0e-3, I_rho=2.0e-3,
                    local_z=np.array([0.0, 1.0, 0.0], dtype=float)),
            ],
            {0: [True, True, True, True, True, True]},
            {3: [-21.2, -21.2, -21.2, 5.0, 0.0, 0.0]},  # axial + tip moment
        ],

        # 5) Moderate cantilever (3 nodes), circular, rotated ~35° about [1,1,0],
        #    with lateral + small tip moment
        [
            np.array([[0.0, 0.0, 0.0],
                    [1.23, 0.88, 2.12],
                    [2.46, 1.76, 4.24]], dtype=float),
            [
                dict(node_i=0, node_j=1, E=8.0e4, nu=0.30,
                    A=np.pi*0.18**2,
                    I_y=np.pi*0.18**4/4, I_z=np.pi*0.18**4/4,
                    J=np.pi*0.18**4/2, I_rho=np.pi*0.18**4/2,
                    local_z=np.array([1.0, 0.0, 0.0], dtype=float)),
                dict(node_i=1, node_j=2, E=8.0e4, nu=0.30,
                    A=np.pi*0.18**2,
                    I_y=np.pi*0.18**4/4, I_z=np.pi*0.18**4/4,
                    J=np.pi*0.18**4/2, I_rho=np.pi*0.18**4/2,
                    local_z=np.array([1.0, 0.0, 0.0], dtype=float)),
            ],
            {0: [True, True, True, True, True, True]},
            {2: [-10.6, -10.6, -10.6, 0.0, 0.0, 0.3]},  # axial + lateral + moment
        ],
    ]
    test_cases = [{"test_code": test_euler_buckling_cantilever_circular_param_sweep, "expected_failures": [elastic_critical_load_analysis_frame_3D_all_zeros, elastic_critical_load_analysis_frame_3D_all_ones]},
                  {"test_code": test_orientation_invariance_cantilever_buckling_rect_section, "expected_failures": [elastic_critical_load_analysis_frame_3D_all_zeros, elastic_critical_load_analysis_frame_3D_all_ones]},
                  {"test_code": test_cantilever_euler_buckling_mesh_convergence, "expected_failures": [elastic_critical_load_analysis_frame_3D_all_zeros, elastic_critical_load_analysis_frame_3D_all_ones]}
                  ]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
