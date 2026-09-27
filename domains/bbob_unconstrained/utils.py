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
    prob['domain'] = 'bbob_unconstrained'
    prob['problem_id'] = str(row.get('problem_id', ''))
    prob['dimension'] = int(row.get('dimension', 10))
    prob['instance'] = int(row.get('instance', 1))
    
    # 添加历史信息路径（不直接嵌入内容，告诉task agent位置）
    if prev_gen_info is not None:
        prob['previous_generation_info'] = {
            'generation': prev_gen_info.get('generation', 'unknown'),
            'total_problems': prev_gen_info.get('total_problems', 0),
            'history_files': {
                'generate_log': '/hada/outputs/{run_id}/gen_{gen}/generate.log',
                'report': '/hada/outputs/{run_id}/gen_{gen}/report.json',
                'predictions': '/hada/outputs/{run_id}/gen_{gen}/bbob_unconstrained_eval/agent_evals/predictions.csv',
                'patch': '/hada/outputs/{run_id}/gen_{gen}/bbob_unconstrained_eval/agent_evals/all_patch.diff',
            },
            'note': 'Use editor tool to view these files for previous generation analysis'
        }
    
    # 添加特殊的 task instruction，告诉 task agent 使用工具修改代码
    prob['task_instruction'] = f"""You are an expert optimization algorithm researcher. Your task is to improve the evolutionary algorithm to solve BBOB (Black-Box Optimization Benchmark) problems.

IMPORTANT: You MUST use the `editor` tool to actually modify the code files. Do NOT just describe changes - you must USE the editor tool with command='str_replace' to make the changes.

## Codebase Structure (/hada/metabbo/):

The codebase has two main layers:

### 1. Evolutionary Algorithm Layer (ec_algorithm.py)
- **ec_algorithm.py**: The main evolutionary algorithm implementation. Contains the optimizer class that handles population initialization, iteration loop, solution evaluation, and result tracking. This is where you implement the core search algorithm (PSO, DE, CMA-ES, etc.).
- **param_controller.py**: Parameter controller interface. Defines the abstract interface for dynamic parameter adjustment. You can implement custom controllers here.

### 2. Meta-Learning Layer (meta_learning.py, meta_learning_env.py, train_meta_learning.py)
- **meta_learning.py**: Meta-learning controller that dynamically adjusts algorithm parameters during optimization. Uses a neural network to map observations to parameter values. Supports both continuous and discrete action spaces.
- **meta_learning_env.py**: Environment wrapper for training the meta-learning controller. Provides a gym-like interface with reset/step methods.
- **train_meta_learning.py**: Training scripts for the meta-learning controller. Contains replay buffer and training logic.

## CRITICAL: Try Different Evolutionary Algorithms!

Previous generations have mostly made small PSO parameter tweaks (adjusting w, c1, c2, adding turbulence, changing initialization). To achieve significant improvements, you should try fundamentally different approaches.

**Consider replacing or substantially modifying the algorithm:**
- **DE (Differential Evolution)**: Excellent for continuous optimization, strong exploration
- **CMA-ES**: State-of-the-art for black-box optimization, adapts covariance matrix
- **SHADE/L-SHADE**: Success-History based Adaptive DE, CEC competition winner
- **JADE**: Adaptive DE with optional external archive
- **ES (Evolution Strategies)**: Simple but effective, uses mutation + selection
- Or any other evolutionary algorithm you think is suitable

**You can also try:**
- Hybrid algorithms combining multiple approaches
- Multi-population / island model with migration
- Archive-based selection mechanisms
- Different initialization strategies, boundary handling, diversity maintenance

**Meta-Learning Layer:**
- Try different neural network architectures (deeper, wider, different activations)
- Change the observation space (what information the controller sees)
- Change the action space (what parameters the controller outputs)
- Try different training strategies (reward design, learning rate, exploration)
- Consider non-DQN approaches (PPO, Bayesian Optimization, etc.)

## File Modification Guide:

To modify a file, use the editor tool in this format:
<json>
{{
    "tool_name": "editor",
    "tool_input": {{
        "command": "str_replace",
        "path": "/hada/metabbo/ec_algorithm.py",
        "old_str": "... the exact code to replace ...",
        "new_str": "... the new code ..."
    }}
}}
</json>

## EVALUATION COUNTING:
- Evaluations are tracked via the `evals` variable in `ec_algorithm.py`
- Each time you call `evaluate_problem(problem, cand)`, you MUST increment `evals += 1`
- The optimization loop runs while `evals < self.max_evals`
- If you change the algorithm, ensure `evals` is incremented correctly for every function evaluation

NOTE: The scoring is handled externally. Do NOT modify any scoring or evaluation code. Only modify the algorithm implementation.

⚠️ IMPORTANT: Before implementing changes, READ the domain code at `/hada/domains/bbob_unconstrained/dqn_de_util.py` to understand the exact scoring logic, evaluation counting, and return format requirements.
"""
    return prob