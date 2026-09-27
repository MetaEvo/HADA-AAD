import sys
sys.path.insert(0, '/hada')
from config import MODEL_NAME as MODEL

QUESTION_ID = "problem_id"

def format_input_dict(row, prev_gen_info=None):
    """Convert a csv row into the input dict for TaskAgent.forward.

    Expected columns in `row`: `problem_id`,`dimension`,`instance`.
    Returns a dict with keys: `problem_id`, `dimension`, `instance`, and
    `task_instruction` to tell the task agent what to do.
    """
    prob = {}
    prob['domain'] = 'bbob_constrained'
    prob['problem_id'] = str(row.get('problem_id', ''))
    prob['dimension'] = int(row.get('dimension', 10))
    prob['instance'] = int(row.get('instance', 1))
    prob['constraint_num'] = int(row.get('constraint_num', 1))
    
    # 添加历史信息路径（不直接嵌入内容，告诉task agent位置）
    if prev_gen_info is not None:
        prob['previous_generation_info'] = {
            'generation': prev_gen_info.get('generation', 'unknown'),
            'total_problems': prev_gen_info.get('total_problems', 0),
            'history_files': {
                'generate_log': '/hada/outputs/{run_id}/gen_{gen}/generate.log',
                'report': '/hada/outputs/{run_id}/gen_{gen}/report.json',
                'predictions': '/hada/outputs/{run_id}/gen_{gen}/bbob_constrained_eval/agent_evals/predictions.csv',
                'patch': '/hada/outputs/{run_id}/gen_{gen}/bbob_constrained_eval/agent_evals/all_patch.diff',
            },
            'note': 'Use editor tool to view these files for previous generation analysis'
        }
    
    # 添加特殊的 task instruction，告诉 task agent 使用工具修改代码以支持约束优化
    prob['task_instruction'] = f"""You are an expert optimization algorithm researcher. Your task is to modify the evolutionary algorithm to solve BBOB CONSTRAINED optimization problems.

IMPORTANT: You MUST use the `editor` tool to actually modify the code files. Do NOT just describe changes - you must USE the editor tool with command='str_replace' to make the changes.

## Codebase Structure (/hada/metabbo/):

The codebase has two main layers:

### 1. Evolutionary Algorithm Layer (ec_algorithm.py)
- **ec_algorithm.py**: The main evolutionary algorithm implementation. Contains the optimizer class that handles population initialization, iteration loop, solution evaluation, and result tracking. This is where you implement the core search algorithm and constraint handling.
- **param_controller.py**: Parameter controller interface. Defines the abstract interface for dynamic parameter adjustment.

### 2. Meta-Learning Layer (meta_learning.py, meta_learning_env.py, train_meta_learning.py)
- **meta_learning.py**: Meta-learning controller that dynamically adjusts algorithm parameters during optimization. Uses a neural network to map observations to parameter values.
- **meta_learning_env.py**: Environment wrapper for training the meta-learning controller. Provides a gym-like interface.
- **train_meta_learning.py**: Training scripts for the meta-learning controller.

## CRITICAL: Try Different Evolutionary Algorithms!

Previous generations have mostly made small parameter tweaks. To achieve significant improvements, you should try fundamentally different approaches.

**Consider replacing or substantially modifying the algorithm:**
- **DE (Differential Evolution)**: Excellent for continuous optimization, can be adapted for constraints
- **CMA-ES**: State-of-the-art for black-box optimization, can handle constraints via penalty or repair
- **SHADE/L-SHADE**: Adaptive DE with strong performance, can incorporate constraint handling
- **ε-constrained DE**: Uses relaxation parameter for constraint handling
- Or any other evolutionary algorithm with constraint handling capabilities

**Constraint handling techniques to consider:**
- Penalty methods (static, dynamic, adaptive)
- Feasibility rules (compare feasibility first, then fitness)
- Repair methods (project infeasible solutions to feasible region)
- Multi-objective transformation (treat constraints as additional objectives)

**Meta-Learning Layer:**
- Try different neural network architectures
- Change the observation space to include constraint violation information
- Change the action space to output constraint-specific parameters (e.g., penalty weights)
- Try different training strategies

CRITICAL: Before modifying any file, you MUST first view the current content of the file using the `editor` tool with command='view'. The code may have been modified by previous agents.

## COCOEX Constrained Problem API Usage:

For BBOB constrained problems, the cocoex problem object has the following API:

```python
import cocoex
import numpy as np

# Create a constrained problem
suite = cocoex.Suite("bbob-constrained", "", "")
problem = suite.get_problem_by_function_dimension_instance(function_id, dimension, instance_id)

# 1. Get objective function value (returns float)
f_value = problem(x)  # x is a numpy array or list

# 2. Get constraint values (returns numpy array)
constraints = problem.constraint(x)  # Returns array of constraint values

# 3. Check if constraints are satisfied (all constraints <= 0)
is_feasible = np.all(constraints <= 0)

# 4. Problem properties
problem.dimension  # Number of dimensions
problem.number_of_constraints  # Number of constraints
problem.lower_bounds  # Lower bounds array
problem.upper_bounds  # Upper bounds array
problem.initial_solution  # Initial solution (if available)
```

## Key Points for Constrained Optimization:

1. **Constraint Satisfaction**: A solution is feasible if ALL constraint values <= 0
2. **Common Approaches**:
   - Penalty method: Add large penalty to objective when constraints are violated
   - Repair method: Project infeasible solutions back to feasible region
   - Constraint-aware EA: Modify search operators to respect constraints
3. **Evaluation**: Only `problem(x)` counts as evaluation. `problem.constraint(x)` does NOT consume evaluation budget.

## CRITICAL Output Requirements:
- Algorithm must return `gbest` (best solution found) and `final_pbest` (list of all particle/solution positions)
- DO NOT change `evaluate_problem(problem, x)` signature - it MUST return a **scalar float**
- External scoring evaluates the final `gbest` against the known optimum

## File Modification Guide:

To modify a file, use the editor tool in this format:
<json>
{{
    "tool_name": "editor",
    "tool_input": {{
        "command": "str_replace",
        "path": "/hada/metabbo/ec_algorithm.py",
        "old_str": "... the EXACT current code to replace (copy from view output) ...",
        "new_str": "... the new code ..."
    }}
}}
</json>

## EVALUATION COUNTING:
- Evaluations are tracked via the `evals` variable in `ec_algorithm.py`
- Each time you call `evaluate_problem(problem, cand)`, you MUST increment `evals += 1`
- The optimization loop runs while `evals < self.max_evals`
- IMPORTANT: Only `problem(x)` counts as evaluation. `problem.constraint(x)` does NOT consume evaluation budget. You can call `problem.constraint(x)` freely to check feasibility without worrying about evaluation cost.

MANDATORY REQUIREMENT: You MUST make at least ONE meaningful improvement. Simply viewing the code without making changes is NOT acceptable.

NOTE: The scoring is handled externally. Do NOT modify any scoring or evaluation code.

⚠️ IMPORTANT: Before implementing changes, READ the domain code at `/hada/domains/bbob_constrained/dqn_de_util.py` to understand the exact scoring logic, constraint handling, evaluation counting, and return format requirements.

"""
    return prob