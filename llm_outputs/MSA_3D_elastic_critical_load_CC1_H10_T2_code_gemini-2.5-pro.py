import numpy as np
import scipy
from typing import Callable, Optional, Sequence
import pytest
def MSA_3D_elastic_critical_load_CC1_H10_T2(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
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
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain:
            'node_i', 'node_j' : int
                End node indices (0-based).
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
                Material and geometric properties.
            'local_z' : array-like of shape (3,), optional
                Unit vector in global coordinates defining the local z-axis orientation.
                Must not be parallel to the beam axis. If None, a default is chosen.
                Default local_z = global z, unless the beam lies along global z — then default local_z = global y.
    boundary_conditions : dict[int, Sequence[int]]
        Maps node index to a 6-element iterable of 0 (free) or 1 (fixed) values.
        Omitted nodes are assumed to have all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        Maps node index to a 6-element array of applied loads:
        [Fx, Fy, Fz, Mx, My, Mz]. Omitted nodes are assumed to have zero loads.
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
    Helper Functions (used here)
    ----------------------------
        Returns the 12x12 local geometric stiffness matrix with torsion-bending coupling for a 3D Euler-Bernoulli beam element.
    Raises
    ------
    ValueError
        Propagated from called routines if:
    Notes
    -----
      if desired (e.g., by max absolute translational DOF).
      the returned mode can depend on numerical details.
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    """
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    K = np.zeros((n_dof, n_dof))
    P = np.zeros(n_dof)
    element_data_cache = []
    for el in elements:
        i, j = el['node_i'], el['node_j']
        node_i_coords, node_j_coords = node_coords[i], node_coords[j]
        E, nu, A, Iy, Iz, J = el['E'], el['nu'], el['A'], el['I_y'], el['I_z'], el['J']
        G = E / (2 * (1 + nu))
        vec_ij = node_j_coords - node_i_coords
        L = np.linalg.norm(vec_ij)
        if L < 1e-9:
            raise ValueError(f"Element between nodes {i} and {j} has zero length.")
        x_local = vec_ij / L
        local_z_vec = el.get('local_z')
        if local_z_vec is not None:
            z_global_ref = np.array(local_z_vec, dtype=float)
        else:
            if np.allclose(np.abs(x_local), [0, 0, 1]):
                z_global_ref = np.array([0., 1., 0.])
            else:
                z_global_ref = np.array([0., 0., 1.])
        if np.linalg.norm(np.cross(z_global_ref, x_local)) < 1e-9:
             raise ValueError(f"local_z vector for element between nodes {i} and {j} is parallel to the beam axis.")
        y_local = np.cross(z_global_ref, x_local)
        y_local /= np.linalg.norm(y_local)
        z_local = np.cross(x_local, y_local)
        rot_mat = np.array([x_local, y_local, z_local])
        T = np.zeros((12, 12))
        for k in range(4):
            T[3*k:3*(k+1), 3*k:3*(k+1)] = rot_mat
        EA_L = E * A / L
        GJ_L = G * J / L
        EIz_L = E * Iz / L
        EIz_L2 = E * Iz / L**2
        EIz_L3 = E * Iz / L**3
        EIy_L = E * Iy / L
        EIy_L2 = E * Iy / L**2
        EIy_L3 = E * Iy / L**3
        k_e = np.array([
            [EA_L, 0, 0, 0, 0, 0, -EA_L, 0, 0, 0, 0, 0],
            [0, 12*EIz_L3, 0, 0, 0, 6*EIz_L2, 0, -12*EIz_L3, 0, 0, 0, 6*EIz_L2],
            [0, 0, 12*EIy_L3, 0, -6*EIy_L2, 0, 0, 0, -12*EIy_L3, 0, -6*EIy_L2, 0],
            [0, 0