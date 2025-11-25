def linear_uniform_mesh_1D(x_min: float, x_max: float, num_elements: int) -> np.ndarray:
    """
    Generate a 1D linear mesh.
    Parameters:
        x_min (float): Start coordinate of the domain.
        x_max (float): End coordinate of the domain.
        num_elements (int): Number of linear elements.
    Returns:
        (node_coords, element_connectivity):
              (shape: [num_elements, 2]) with node indices per element
    """
    import numpy as np
    try:
        x_min = float(x_min)
        x_max = float(x_max)
    except Exception:
        raise TypeError('x_min and x_max must be real numbers')
    if not (np.isfinite(x_min) and np.isfinite(x_max)):
        raise ValueError('x_min and x_max must be finite')
    try:
        ne = int(num_elements)
    except Exception:
        raise TypeError('num_elements must be an integer')
    if ne != num_elements:
        raise TypeError('num_elements must be an integer')
    if ne < 1:
        raise ValueError('num_elements must be >= 1')
    if x_min == x_max:
        raise ValueError('Domain length must be non-zero')
    node_coords = np.linspace(x_min, x_max, ne + 1)
    start = np.arange(ne, dtype=int)
    element_connectivity = np.column_stack((start, start + 1))
    return (node_coords, element_connectivity)