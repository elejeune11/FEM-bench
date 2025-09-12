### Function Correctness (✓ = Match)

| Task                                               | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_stiffness_matrix_linear_elastic_3D | ✓            | ✓               | ✓              | ×            | ✓        |
| beam_transformation_matrix_3D                      | ×            | ×               | ×              | ×            | ✓        |
| compute_local_element_loads_beam_3D                | ✓            | ✓               | ✓              | ×            | ✓        |
| element_stiffness_linear_elastic_1D                | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_solve                                       | ✓            | ×               | ×              | ✓            | ✓        |
| linear_uniform_mesh_1D                             | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam             | ✓            | ×               | ×              | ×            | ✓        |
| local_geometric_stiffness_matrix_3D_beam           | ×            | ×               | ×              | ×            | ×        |
| solve_linear_elastic_1D_self_contained             | ✓            | ✓               | ×              | ×            | ✓        |
| solve_linear_elastic_frame_3D                      | ✓            | ✓               | ✓              | ×            | ✓        |
| solve_linear_elastic_frame_3D_self_contained       | ✓            | ×               | ×              | ×            | ×        |
| Total                                              | 9/11         | 6/11            | 5/11           | 2/11         | 9/11     |

### Joint Test Success Rate (%)

| Task                                               | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_stiffness_matrix_linear_elastic_3D | 100.0%       | 0.0%            | 100.0%         | 0.0%         | 0.0%     |
| beam_transformation_matrix_3D                      | 33.3%        | 66.7%           | 33.3%          | 33.3%        | 33.3%    |
| compute_local_element_loads_beam_3D                | 0.0%         | 75.0%           | –              | 0.0%         | 75.0%    |
| element_stiffness_linear_elastic_1D                | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_solve                                       | 0.0%         | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| linear_uniform_mesh_1D                             | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam             | 0.0%         | –               | 0.0%           | 100.0%       | 50.0%    |
| local_geometric_stiffness_matrix_3D_beam           | 50.0%        | –               | 50.0%          | 0.0%         | 50.0%    |
| solve_linear_elastic_1D_self_contained             | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| solve_linear_elastic_frame_3D                      | 100.0%       | 100.0%          | 50.0%          | 0.0%         | 0.0%     |
| solve_linear_elastic_frame_3D_self_contained       | 66.7%        | –               | 33.3%          | 33.3%        | 33.3%    |
| Avg Joint Success %                                | 50.0%        | 40.2%           | 47.0%          | 42.4%        | 44.7%    |

