### Function Correctness (✓ = Match)

| Task                                     | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:-----------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| beam_transformation_matrix_3D            | ×            | ×               | ×              | ×            | ✓        |
| element_stiffness_linear_elastic_1D      | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_uniform_mesh_1D                   | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam   | ✓            | ×               | ×              | ×            | ✓        |
| local_geometric_stiffness_matrix_3D_beam | ×            | ×               | ×              | ×            | ×        |
| solve_linear_elastic_1D_self_contained   | ✓            | ✓               | ×              | ×            | ✓        |
| solve_linear_elastic_frame_3D            | ✓            | ×               | ×              | ×            | ×        |
| Total                                    | 5/7          | 3/7             | 2/7            | 1/7          | 5/7      |

### Joint Test Success Rate (%)

| Task                                     | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:-----------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| beam_transformation_matrix_3D            | 33.3%        | 66.7%           | 33.3%          | 33.3%        | 33.3%    |
| element_stiffness_linear_elastic_1D      | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                   | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam   | 0.0%         | –               | 0.0%           | 100.0%       | 50.0%    |
| local_geometric_stiffness_matrix_3D_beam | 50.0%        | –               | 50.0%          | 0.0%         | 50.0%    |
| solve_linear_elastic_1D_self_contained   | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| solve_linear_elastic_frame_3D            | 33.3%        | –               | 66.7%          | –            | 33.3%    |
| Avg Joint Success %                      | 45.2%        | 38.1%           | 57.1%          | 61.9%        | 59.5%    |

