def FEM_2D_quad8_element_distributed_load_CC0_H0_T0(face: int, node_coords: np.ndarray, traction: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    if num_gauss_pts == 1:
        gauss_pts = [0.0]
        gauss_wts = [2.0]
    elif num_gauss_pts == 2:
        gauss_pts = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gauss_wts = [1.0, 1.0]
    elif num_gauss_pts == 3:
        gauss_pts = [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)]
        gauss_wts = [5 / 9, 8 / 9, 5 / 9]
    else:
        raise ValueError('num_gauss_pts must be 1, 2, or 3')
    face_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    (start_idx, mid_idx, end_idx) = face_nodes[face]
    r_elem = np.zeros(16)
    for (s, w) in zip(gauss_pts, gauss_wts):
        N1 = 0.5 * s * (s - 1)
        N5 = 1 - s * s
        N2 = 0.5 * s * (s + 1)
        (x_start, y_start) = node_coords[start_idx]
        (x_mid, y_mid) = node_coords[mid_idx]
        (x_end, y_end) = node_coords[end_idx]
        dx_ds = N1 * x_start + N5 * x_mid + N2 * x_end
        dy_ds = N1 * y_start + N5 * y_mid + N2 * y_end
        J = np.sqrt(dx_ds ** 2 + dy_ds ** 2)
        load_factor = traction[0] * N1 * J * w
        r_elem[2 * start_idx] += load_factor
        r_elem[2 * start_idx + 1] += traction[1] * N1 * J * w
        load_factor = traction[0] * N5 * J * w
        r_elem[2 * mid_idx] += load_factor
        r_elem[2 * mid_idx + 1] += traction[1] * N5 * J * w
        load_factor = traction[0] * N2 * J * w
        r_elem[2 * end_idx] += load_factor
        r_elem[2 * end_idx + 1] += traction[1] * N2 * J * w
    return r_elem