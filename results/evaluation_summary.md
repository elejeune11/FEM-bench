### Function Correctness (✓ = Match)

| Task                                               | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_stiffness_matrix_linear_elastic_3D | ×            | ×               | ×              | ×            | ×        |
| beam_transformation_matrix_3D                      | ×            | ×               | ×              | ×            | ✓        |
| element_stiffness_linear_elastic_1D                | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_uniform_mesh_1D                             | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam             | ✓            | ×               | ×              | ×            | ✓        |
| local_geometric_stiffness_matrix_3D_beam           | ×            | ×               | ×              | ×            | ×        |
| solve_linear_elastic_1D_self_contained             | ✓            | ✓               | ×              | ×            | ✓        |
| solve_linear_elastic_frame_3D                      | ✓            | ×               | ×              | ×            | ×        |
| Total                                              | 5/8          | 3/8             | 2/8            | 1/8          | 5/8      |

### Joint Test Success Rate (%)

| Task                                               | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| assemble_global_stiffness_matrix_linear_elastic_3D | 100.0%       | 0.0%            | 0.0%           | 0.0%         | 0.0%     |
| beam_transformation_matrix_3D                      | 33.3%        | 66.7%           | 33.3%          | 33.3%        | 33.3%    |
| element_stiffness_linear_elastic_1D                | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                             | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam             | 0.0%         | –               | 0.0%           | 100.0%       | 50.0%    |
| local_geometric_stiffness_matrix_3D_beam           | 50.0%        | –               | 50.0%          | 0.0%         | 50.0%    |
| solve_linear_elastic_1D_self_contained             | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| solve_linear_elastic_frame_3D                      | 33.3%        | –               | 66.7%          | –            | 33.3%    |
| Avg Joint Success %                                | 52.1%        | 33.3%           | 50.0%          | 54.2%        | 52.1%    |

