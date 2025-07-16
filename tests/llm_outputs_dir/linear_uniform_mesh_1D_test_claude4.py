def test_basic_mesh_creation(fcn):
   """Test basic mesh creation with simple parameters."""
   x_min = 0.0
   x_max = 1.0
   num_elements = 4
   
   node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
   
   assert node_coords.shape == (5,)
   assert element_connectivity.shape == (4, 2)
   
   expected_coords = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   assert np.allclose(node_coords, expected_coords)
   
   expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
   assert np.array_equal(element_connectivity, expected_connectivity)
   
   assert node_coords[0] == x_min
   assert node_coords[-1] == x_max

def test_single_element_mesh(fcn):
   """Test edge case with only one element."""
   x_min = -1.0
   x_max = 3.0
   num_elements = 1
   
   node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
   
   assert node_coords.shape == (2,)
   assert element_connectivity.shape == (1, 2)
   
   expected_coords = np.array([-1.0, 3.0])
   assert np.allclose(node_coords, expected_coords)
   
   expected_connectivity = np.array([[0, 1]])
   assert np.array_equal(element_connectivity, expected_connectivity)
   
   assert node_coords[0] == x_min
   assert node_coords[1] == x_max