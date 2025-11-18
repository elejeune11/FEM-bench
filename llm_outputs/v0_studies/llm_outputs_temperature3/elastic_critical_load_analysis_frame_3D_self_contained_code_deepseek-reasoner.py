import numpy as np
import scipy
from typing import Optional, Sequence
import pytest
def elastic_critical_load_analysis_frame_3D_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
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
                Start node index (0-based).
                End node index (0-based).
          Material
          --------
                Young's modulus (used in axial, bending, and torsion terms).
                Poisson's ratio (used in torsion only, per your stiffness routine).
          Section (local axes y,z about the beam's local x)
          -----------------------------------------------
                Cross-sectional area.
                Second moment of area about local y.
                Second moment of area about local z.
                Torsional constant (for elastic/torsional stiffness).
                Polar moment about the local x-axis used by the geometric stiffness
                with torsion-bending coupling (see your geometric K routine).
          Orientation
          -----------
                Provide a 3-vector giving the direction of the element's local z-axis to 
                disambiguate the local triad used in the 12x12 transformation; if set to `None`, 
                a default convention will be applied to construct the local axes.
    boundary_conditions : dict
        Dictionary mapping node index -> boundary condition specification. Each
        node's specification can be provided in either of two forms:
          the DOF is constrained (fixed).
          at that node are fixed.
        All constrained DOFs are removed from the free set. It is the caller's
        responsibility to supply constraints sufficient to eliminate rigid-body
        modes.
    nodal_loads : dict[int, Sequence[float]]
        Mapping from node index → length-6 vector of load components applied at
        that node in the **global** DOF order `[F_x, F_y, F_z, M_x, M_y, M_z]`.
        Consumed by `assemble_global_load_vector_linear_elastic_3D` to form `P`.
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
      `[u_x, u_y, u_z, θ_x, θ_y, θ_z]`.
      represented via `K_g` assembled at the reference load state, not via a full
      nonlinear equilibrium/path-following analysis.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
    """
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = assemble_global_stiffness_linear_elastic_3D(node_coords, elements)
    P = assemble_global_load_vector_linear_elastic_3D(node_coords, nodal_loads)
    constrained_dofs = set()
    for node_idx, bc_spec in boundary_conditions.items():
        if len(bc_spec) == 6 and all(isinstance(x, bool) for x in bc_spec):
            for dof_idx, is_constrained in enumerate(bc_spec):
                if is_constrained:
                    constrained_dofs.add(6 * node_idx + dof_idx)
        else:
            for dof_idx in bc_spec:
                constrained_dofs.add(6 * node_idx + dof_idx)
    free_dofs = sorted(set(range(n_dofs)) - constrained_dofs)
    if len(free_dofs) == 0:
        raise ValueError("No free DOFs remain after applying boundary conditions")
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    P_f = P[free_dofs]
    try:
        u_f = np.linalg.solve(K_ff, P_f)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Linear static solution failed: {e}")
    u = np.zeros(n_dofs)
    u[free_dofs] = u_f
    K_g = assemble_global_geometric_stiffness_3D(node_coords, elements, u)
    K_g_ff = K_g[np.ix_(free_dofs, free_dofs)]
    try:
        eigvals, eigvecs = scipy.linalg.eig(K_ff, -K_g_ff)
    except (scipy.linalg.LinAlgError, ValueError) as e:
        raise ValueError(f"Eigenproblem solution failed: {e}")
    real_mask = np.abs(eigvals.imag) < 1e-12
    if not np.any(real_mask):
        raise ValueError("No real eigenvalues found in buckling analysis")
    eigvals_real = eigvals[real_mask].real
    eigvecs_real = eigvecs[:, real_mask].real
    positive_mask = eigvals_real > 1e-12
    if not np.any(positive_mask):
        raise ValueError("No positive eigenvalues found in buckling analysis")
    eigvals_positive = eigvals_real[positive_mask]
    eigvecs_positive = eigvecs_real[:, positive_mask]
    min_idx = np.argmin(eigvals_positive)
    elastic_critical_load_factor = eigvals_positive[min_idx]
    phi_f = eigvecs_positive[:, min_idx]
    deformed_shape_vector = np.zeros(n_dofs)
    deformed_shape_vector[free_dofs] = phi_f
    return elastic_critical_load_factor, deformed_shape_vector
def assemble_global_stiffness_linear_elastic_3D(node_coords, elements):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        i, j = elem['node_i'], elem['node_j']
        E, nu = elem['E'], elem['nu']
        A, Iy, Iz, J, I_rho = elem['A'], elem['Iy'], elem['Iz'], elem['J'], elem['I_rho']
        local_z = elem.get('local_z', None)
        k_local = compute_local_stiffness_matrix_3D(E, nu, A, Iy, Iz, J, elem)
        T = compute_transformation_matrix_3D(node_coords[i], node_coords[j], local_z)
        k_global = T.T @ k_local @ T
        dofs = np.concatenate([6*i + np.arange(6), 6*j + np.arange(6)])
        K[np.ix_(dofs, dofs)] += k_global
    return K
def assemble_global_load_vector_linear_elastic_3D(node_coords, nodal_loads):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    P = np.zeros(n_dofs)
    for node_idx, load_vec in nodal_loads.items():
        if len(load_vec) != 6:
            raise ValueError(f"Load vector for node {node_idx} must have 6 components")
        start_dof = 6 * node_idx
        P[start_dof:start_dof+6] = load_vec
    return P
def assemble_global_geometric_stiffness_3D(node_coords, elements, displacements):
    n_nodes = node_coords.shape[0]
    n_dofs = 6 * n_nodes
    K_g = np.zeros((n_dofs, n_dofs))
    for elem in elements:
        i, j = elem['node_i'], elem['node_j']
        E, nu = elem['E'], elem['nu']
        A, Iy, Iz, J, I_rho = elem['A'], elem['Iy'], elem['Iz'], elem['J'], elem['I_rho']
        local_z = elem.get('local_z', None)
        elem_disp_global = np.concatenate([
            displacements[6*i:6*i+6],
            displacements[6*j:6*j+6]
        ])
        T = compute_transformation_matrix_3D(node_coords[i], node_coords[j], local_z)
        elem_disp_local = T @ elem_disp_global
        k_g_local = compute_local_geometric_stiffness_matrix_3D(elem, elem_disp_local)
        k_g_global = T.T @ k_g_local @ T
        dofs = np.concatenate([6*i + np.arange(6), 6*j + np.arange(6)])
        K_g[np.ix_(dofs, dofs)] += k_g_global
    return K_g
def compute_local_stiffness_matrix_3D(E, nu, A, Iy, Iz, J, elem):
    L = np.linalg.norm(np.array(elem['node_j_coords']) - np.array(elem['node_i_coords']))
    k_local = np.zeros((12, 12))
    EA_L = E * A / L
    k_local[0, 0] = EA_L
    k_local[6, 6] = EA_L
    k_local[0, 6] = -EA_L
    k_local[6, 0] = -EA_L
    G = E / (2 * (1 + nu))
    GJ_L = G * J / L
    k_local[3, 3] = GJ_L
    k_local[9, 9] = GJ_L
    k_local[3, 9] = -GJ_L
    k_local[9, 3] = -GJ_L
    EIy_L3 = E * Iy / L**3
    k_local[[1, 1, 1, 7, 7, 7, 5, 5, 5, 11, 11, 11], 
            [1, 7, 5, 1, 7, 11, 1, 5, 11, 7, 11, 5]] = [
        12*EIy_L3, -12*EIy_L3, 6*EIy_L3*L,
        -12*EIy_L3, 12*EIy_L3, -6*EIy_L3*L,
        6*EIy_L3*L, 4*EIy_L3*L**2, 2*EIy_L3*L**2,
        -6*EIy_L3*L, 4*EIy_L3*L**2, 2*EIy_L3*L**2
    ]
    k_local[5, 11] = 2*EIy_L3*L**2
    k_local[11, 5] = 2*EIy_L3*L**2
    EIz_L3 = E * Iz / L**3
    k_local[[2, 2, 2, 8, 8, 8, 4, 4, 4, 10, 10, 10], 
            [2, 8, 4, 2, 8, 10, 2, 4, 10, 8, 10, 4]] = [
        12*EIz_L3, -12*EIz_L3, -6*EIz_L3*L,
        -12*EIz_L3, 12*EIz_L3, 6*EIz_L3*L,
        -6*EIz_L3*L, 4*EIz_L3*L**2, 2*EIz_L3*L**2,
        6*EIz_L3*L, 4*EIz_L3*L**2, 2*EIz_L3*L**2
    ]
    k_local[4, 10] = 2*EIz_L3*L**2
    k_local[10, 4] = 2*EIz_L3*L**2
    return k_local
def compute_local_geometric_stiffness_matrix_3D(elem, local_displacements):
    L = np.linalg.norm(np.array(elem['node_j_coords']) - np.array(elem['node_i_coords']))
    P = compute_axial_force(elem, local_displacements)
    k_g_local = np.zeros((12, 12))
    if abs(P) < 1e-12:
        return k_g_local
    P_L = P / L
    k_g_local[[1, 1, 1, 7, 7, 7, 5, 5, 5, 11, 11, 11], 
              [1, 7, 5, 1, 7, 11, 1, 5, 11, 7, 11, 5]] = [
        6/5*P_L, -6/5