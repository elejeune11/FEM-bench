def solve_linear_elastic_frame_3D(node_coords: np.ndarray, elements: Sequence[dict], boundary_conditions: dict[int, Sequence[int]], nodal_loads: dict[int, Sequence[float]]):
    """
    Small-displacement linear-elastic analysis of a 3D frame. Coordinate system follows the right hand rule.
    The condition number of the global stiffness matrix should be checked before solving.
    If the problem is ill-posed based on condition number, return a (6 N, ) zero array for both u and r.
    Parameters
    ----------
    node_coords : (N, 3) float ndarray
        Global coordinates of the N nodes (row i → [x, y, z] of node i, 0-based).
    elements : iterable of dict
        Each dict must contain
            'node_i', 'node_j' : int # end node indices (0-based)
            'E', 'nu', 'A', 'I_y', 'I_z', 'J' : float
            'local_z' : (3,) array | None # optional unit vector for local z
    boundary_conditions : dict[int, Sequence[int]]
        node index → 6-element 0/1 iterable (0 = free, 1 = fixed).
        Omitted nodes ⇒ all DOFs free.
    nodal_loads : dict[int, Sequence[float]]
        node index → 6-element [Fx, Fy, Fz, Mx, My, Mz] (forces (+) and moments).
        Omitted nodes ⇒ zero loads.
    Returns:
    u : (6 N,) ndarray
        Global displacement vector (UX, UY, UZ, RX, RY, RZ for each node in order).
    r : (6 N,) ndarray
        Global force/moment vector with support reactions filled in fixed DOFs.
    """
    N = len(node_coords)
    u = np.zeros((6 * N,))
    r = np.zeros((6 * N,))
    return (u, r)