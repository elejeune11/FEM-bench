### Function Correctness (✓ = Match)

| Task                                     | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:-----------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| beam_transformation_matrix_3D            | ×            | ×               | ×              | ×            | ✓        |
| element_stiffness_linear_elastic_1D      | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_uniform_mesh_1D                   | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam   | ✓            | ×               | ×              | ×            | ✓        |
| local_geometric_stiffness_matrix_3D_beam | ×            | ×               | ×              | ×            | ×        |
| solve_linear_elastic_1D_self_contained   | ✓            | ✓               | ×              | ×            | ✓        |
| Total                                    | 4/6          | 3/6             | 2/6            | 1/6          | 5/6      |

### Joint Test Success Rate (%)

| Task                                     | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:-----------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| beam_transformation_matrix_3D            | 33.3%        | 66.7%           | 33.3%          | 33.3%        | 33.3%    |
| element_stiffness_linear_elastic_1D      | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                   | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam   | 0.0%         | –               | 0.0%           | 100.0%       | 50.0%    |
| local_geometric_stiffness_matrix_3D_beam | 50.0%        | –               | 50.0%          | 0.0%         | 50.0%    |
| solve_linear_elastic_1D_self_contained   | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| Avg Joint Success %                      | 47.2%        | 44.4%           | 55.6%          | 72.2%        | 63.9%    |

