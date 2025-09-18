import numpy as np
import scipy
from typing import Optional, Sequence
import pytest
def elastic_critical_load_analysis_frame_3D_part_self_contained(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int | bool]], nodal_loads: dict[int, Sequence[float]]):
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
                Poisson's ratio (used in torsion only).
          Section (local axes y,z about the beam's local x)
          -----------------------------------------------
                Cross-sectional area.
                Second moment of area about local y.
                Second moment of area about local z.
                Torsional constant (for elastic/torsional stiffness).
                Polar moment about the local x-axis used by the geometric stiffness
                with torsion–bending coupling.
          Orientation
          -----------
                Provide a 3-vector giving the direction of the element's local z-axis to 
                disambiguate the local triad used in the 12×12 transformation; if `None`, 
                a default convention is applied.
    boundary_conditions : dict
        Dictionary mapping node index -> boundary condition specification. Each
        node's specification can be provided in either of two forms:
          the DOF is constrained (fixed).
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
      `[u_x, u_y, u_z, θ_x, θ_y, θ_z]`.
      represented via `K_g` assembled at the reference load state, not via a full
      nonlinear equilibrium/path-following analysis.
    External Helper Functions (required)
    ------------------------------------
        Local elastic stiffness matrix for a 3D Euler-Bernoulli beam aligned with
        the local x-axis.
        Local geometric stiffness matrix with torsion-bending coupling.
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
    K = np.zeros((n_dofs, n_dofs))
    K_g = np.zeros((n_dofs, n_dofs))
    P = np.zeros(n_dofs)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        E = element['E']
        nu = element['nu']
        A = element['A']
        Iy = element['Iy']
        Iz = element['Iz']
        J = element['J']
        I_rho = element['I_rho']
        local_z = element['local_z']
        coord_i = node_coords[node_i]
        coord_j = node_coords[node_j]
        L = np.linalg.norm(coord_j - coord_i)
        direction_cosines = (coord_j - coord_i) / L
        k_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        if local_z is None:
            local_z = np.array([0, 0, 1])
        local_y = np.cross(local_z, direction_cosines)
        local_y /= np.linalg.norm(local_y)
        local_z = np.cross(direction_cosines, local_y)
        T = np.zeros((12, 12))
        T[0:3, 0:3] = T[6:9, 6:9] = np.vstack((direction_cosines, local_y, local_z)).T
        T[3:6, 3:6] = T[9:12, 9:12] = T[0:3, 0:3]
        k_global = T.T @ k_local @ T
        dof_indices = np.array([6 * node_i, 6 * node_i + 1, 6 * node_i + 2, 6 * node_i + 3, 6 * node_i + 4, 6 * node_i + 5,
                                6 * node_j, 6 * node_j + 1, 6 * node_j + 2, 6 * node_j + 3, 6 * node_j + 4, 6 * node_j + 5])
        for i in range(12):
            for j in range(12):
                K[dof_indices[i], dof_indices[j]] += k_global[i, j]
    for node, load in nodal_loads.items():
        P[6 * node:6 * node + 6] = load
    free_dofs = np.arange(n_dofs)
    for node, bc in boundary_conditions.items():
        if isinstance(bc, Sequence) and all(isinstance(x, bool) for x in bc):
            constrained_dofs = [6 * node + i for i, constrained in enumerate(bc) if constrained]
        elif isinstance(bc, Sequence) and all(isinstance(x, int) for