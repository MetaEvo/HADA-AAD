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
    
    # 添加特殊的 task instruction，告诉 task agent 使用工具修改 PSO 代码
    prob['task_instruction'] = f"""You are an expert optimization algorithm researcher. Your task is to improve the PSO (Particle Swarm Optimization) algorithm to solve BBOB (Black-Box Optimization Benchmark) problems.

IMPORTANT: You MUST use the `editor` tool to actually modify the code files. Do NOT just describe changes - you must USE the editor tool with command='str_replace' to make the changes.

## Available Files in /hada/metabbo/:

- `pso.py` - The main PSO algorithm implementation. Contains particle swarm optimization logic including velocity/position updates, fitness evaluation, and the optimization loop. It integrates with a parameter controller that can dynamically adjust PSO parameters (w, c1, c2) during execution.
- `dqn_controller.py` - A Deep Q-Network based controller that outputs PSO parameters at each iteration. It takes the current optimization state as observation and learns to output parameter values through reinforcement learning training.
- `param_controller.py` - An interface/abstraction layer for parameter control. The DQN controller implements this interface.
- `train_dqn_simple.py` / `train_dqn.py` - Scripts for training the DQN controller through interaction with the optimization process.

## Domain Goal:

The BBOB unconstrained domain consists of 24 single-objective black-box optimization problems with various characteristics (separable, ill-conditioned, multimodal, etc.). The goal is to find the global optimum with limited evaluation budget.

The current implementation uses a DQN-PSO hybrid where the DQN controller dynamically adjusts PSO parameters during optimization. Performance is measured by how close the algorithm gets to the known optimum within the evaluation budget (Relative Improvement score).

## Important Notes:

- The DQN controller's output completely overrides the default PSO parameter values during optimization.
- The DQN is trained through interaction with the optimization process, receiving rewards based on improvement.

## How to Modify:

To modify a file, use the editor tool in this format:
<json>
{{
    "tool_name": "editor",
    "tool_input": {{
        "command": "str_replace",
        "path": "/hada/metabbo/pso.py",
        "old_str": "... the exact code to replace ...",
        "new_str": "... the new code ..."
    }}
}}
</json>

MANDATORY: You MUST make at least one meaningful change to files in `/hada/metabbo/`. Viewing code without modification is NOT acceptable.

NOTE: The scoring is handled externally. Do NOT modify any scoring or evaluation code.

⚠️ IMPORTANT: Before implementing changes, READ the domain code at `/hada/domains/bbob_unconstrained/dqn_de_util.py` to understand the exact scoring logic, evaluation counting, and return format requirements.

"""
    return prob