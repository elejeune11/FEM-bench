import numpy as np
from typing import Callable, Optional, Sequence
import pytest
def MSA_3D_assemble_global_geometric_stiffness_CC1_H4_T2(node_coords: np.ndarray, elements: Sequence[dict], u_global: np.ndarray) -> np.ndarray:
    """
    Assemble the global geometric (initial-stress) stiffness matrix K_g for a 3D frame
    under a given global displacement state.
    Each 2-node Euler–Bernoulli beam contributes a 12×12 local geometric stiffness
    matrix k_g^local that depends on the element length and the internal end
    force/moment resultants induced by the current displacement state. The local
    matrix is then mapped to global coordinates with a 12×12 direction-cosine
    transformation Γ and scattered into the global K_g.
    Parameters
    ----------
    node_coords : (n_nodes, 3) ndarray of float
        Global Cartesian coordinates [x, y, z] of each node (0-based indexing).
    elements : sequence of dict
        Per-element dictionaries. Required keys per element:
            'nodes' : tuple of int
                Indices of the two nodes (i, j) connected by the element.
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
                Must not be parallel to the beam axis. If None, a default is chosen (see Notes).
    u_global : (6*n_nodes,) ndarray of float
        Global displacement vector with 6 DOF per node in the order
        [u_x, u_y, u_z, θ_x, θ_y, θ_z] for node 0, then node 1, etc.
    Returns
    -------
    K : (6*n_nodes, 6*n_nodes) ndarray of float
        Assembled global geometric stiffness matrix. For conservative loading and
        the standard formulation, K_g is symmetric.
    Notes
    -----
      unless the beam axis is aligned with global z, in which case use the global y-axis.
      The 'local_z' must be unit length and not parallel to the beam axis.
      induced by the supplied displacement state (not external loads). Their local DOF
      ordering is the same as for local displacements:
      [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2] ↔
      [Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i, Fx_j, Fy_j, Fz_j, Mx_j, My_j, Mz_j].
      should be treated as an error by the transformation routine.
    External Dependencies
    ---------------------
    local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2) -> (12,12) ndarray
        Must return the local geometric stiffness using the element length L, section properties, and local end force resultants as shown.
    """
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    K_g = np.zeros((n_dof, n_dof))
    for el in elements:
        i, j = el['nodes']
        p1 = node_coords[i]
        p2 = node_coords[j]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-9:
            raise ValueError(f"Element between nodes {i} and {j} has zero length.")
        local_x = vec / L
        if el.get('local_z') is not None:
            ref_vec = np.asarray(el['local_z'])
            if not np.isclose(np.linalg.norm(ref_vec), 1.0):
                raise ValueError(f"Element {i}-{j}: 'local_z' must be a unit vector.")
            if np.isclose(np.abs(np.dot(local_x, ref_vec)), 1.0):
                raise ValueError(f"Element {i}-{j}: 'local_z' must not be parallel to the beam axis.")
        else:
            global_z = np.array([0., 0., 1.])
            if np.isclose(np.abs(np.dot(local_x, global_z)), 1.0):
                ref_vec = np.array([0., 1., 0.])
            else:
                ref_vec = global_z
        local_y_temp = np.cross(local_x, ref_vec)
        local_y = local_y_temp / np.linalg.norm(local_y_temp)
        local_z = np.cross(local_x, local_y)
        R = np.array([local_x, local_y, local_z]).T
        T = np.zeros((12, 12))
        for k in range(4):
            T[k*3:(k+1)*3, k*3:(k+1)*3] = R
        E, nu = el['E'], el['nu']
        A, Iy, Iz, J = el['A'], el['I_y'], el['I_z'], el['J']
        G = E / (2 * (1 + nu))
        I_rho = Iy + Iz
        k_e = np.zeros((12, 12))
        L2, L3 = L*L, L*L*L
        EAL, GJL = E*A/L, G*J/L
        EIz12_L3, EIz6_L2, EIz4_L, EIz2_L = 12*E*Iz/L3, 6*E*Iz/L2, 4*E*Iz/L, 2*E*Iz/L
        EIy12_L3, EIy6_L2, EIy4_L, EIy2_L = 12*E*Iy/L3, 6*E*Iy/L2, 4*E*Iy/L, 2*E*Iy/L
        k_e[0,0], k_e[6,6], k_e[0,6], k_e[6,0] = EAL, EAL, -EAL, -EAL
        k_e[3,3], k_e[9,9], k_e[3,9], k_e[9,3] = GJL, GJL, -GJL, -GJL
        k_e[1,1], k_e[7,7], k_e[1,7], k_e[7,1] = EIz12_L3, EIz12_L3, -EIz12_L3, -EIz12_L3
        k_e[1,5], k_e[5,1], k_e[1,11], k_e[11,1] = EIz6_L2, EIz6_L2, EIz6_L2, EIz6_L2
        k_e[7,5], k_e[5,7], k_e[7,11], k_e[11,7] = -EIz6_L2, -EIz6_L2, -EIz6_L2, -EIz6_L2
        k_e[5,5], k_e[11,11], k_e[5,11], k_e[11,5] = EIz4_L, EIz4_L, EIz2_L, EIz2_L
        k_e[2,2], k_e[8,8], k_e[2,8], k_e[8,2] = EIy12_L3, EIy12_L3, -EIy12_L3, -EIy12_L3
        k_e[2,4], k_e[4,2], k_e[2,10], k_e[10,2] = -EIy6_L2, -EIy6_L2, -EIy6_L2, -EIy6_L2
        k_e[8,4], k_e[4,8], k_e[8,10], k_e[10,8] =