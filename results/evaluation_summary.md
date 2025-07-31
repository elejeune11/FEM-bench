### Function Correctness (✓ = Match)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | ✓            | ✓               | ✓              | ×            | ✓        |
| linear_uniform_mesh_1D                 | ✓            | ✓               | ✓              | ✓            | ✓        |
| solve_linear_elastic_1D_self_contained | ✓            | ✓               | ×              | ×            | ✓        |
| Total                                  | 3/3          | 3/3             | 2/3            | 1/3          | 3/3      |

### Reference Tests Passed (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| solve_linear_elastic_1D_self_contained | 0.0%         | –               | 50.0%          | 100.0%       | 50.0%    |
| Avg Ref Pass %                         | 66.7%        | 66.7%           | 83.3%          | 100.0%       | 83.3%    |

### Expected Failures Detected (%)

| Task                                   | claude-3-5   | deepseek-chat   | gemini-flash   | gemini-pro   | gpt-4o   |
|:---------------------------------------|:-------------|:----------------|:---------------|:-------------|:---------|
| element_stiffness_linear_elastic_1D    | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| linear_uniform_mesh_1D                 | 100.0%       | 100.0%          | 100.0%         | 100.0%       | 100.0%   |
| solve_linear_elastic_1D_self_contained | 100.0%       | –               | 100.0%         | 100.0%       | 100.0%   |
| Avg Fail Detect %                      | 100.0%       | 66.7%           | 100.0%         | 100.0%       | 100.0%   |

