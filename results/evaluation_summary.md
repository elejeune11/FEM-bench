### Function Correctness (✓ = Match)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_uniform_mesh_1D                 | ✓            | ✓               | ✓              | ✓            | ✓        |
| local_elastic_stiffness_matrix_3D_beam | ✓            | ✓               | ×              | ×            | ✓        |
| solve_linear_elastic_1D_self_contained | ✓            | ✓               | ×              | ×            | ✓        |
| Total                                  | 4/4          | 4/4             | 2/4            | 1/4          | 4/4      |

### Reference Tests Passed (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam | 0.0%         | 0.0%            | 0.0%           | 0.0%         | 100.0%   |
| solve_linear_elastic_1D_self_contained | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| Avg Ref Pass %                         | 50.0%        | 50.0%           | 62.5%          | 75.0%        | 87.5%    |

### Expected Failures Detected (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| local_elastic_stiffness_matrix_3D_beam | 100.0%       | 0.0%            | 100.0%         | 100.0%       | 100.0%   |
| solve_linear_elastic_1D_self_contained | 100.0%       | –               | 100.0%         | 100.0%       | 100.0%   |
| Avg Fail Detect %                      | 100.0%       | 50.0%           | 100.0%         | 100.0%       | 100.0%   |

