import sys

# Import global configuration
sys.path.insert(0, '/hada')
from config import set_global_seed, SEED, MODEL_NAME

# Set global seed at module load time
set_global_seed()

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class HyperAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None, domains=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
            domains (str, optional): Comma-separated list of domains being optimized (e.g., 'metabox_mt'). Defaults to None.
        """
        domain_info = f"\nCURRENT DOMAINS BEING OPTIMIZED: {domains}" if domains else ""
        instruction = f"""You are a Meta Agent that improves the task agent's performance.{domain_info}

## Codebase Structure (/hada/cocoex-example/):

The task agent modifies code in two main layers:

### 1. Evolutionary Algorithm Layer (ec_algorithm.py)
- **ec_algorithm.py**: The main evolutionary algorithm implementation (PSO, DE, CMA-ES, etc.). Contains the optimizer class that handles population initialization, iteration loop, solution evaluation, and result tracking.
- **param_controller.py**: Parameter controller interface. Defines the abstract interface for dynamic parameter adjustment.

### 2. Meta-Learning Layer (meta_learning.py, meta_learning_env.py, train_meta_learning.py)
- **meta_learning.py**: Meta-learning controller that dynamically adjusts algorithm parameters during optimization.
- **meta_learning_env.py**: Environment wrapper for training the meta-learning controller.
- **train_meta_learning.py**: Training scripts for the meta-learning controller.

ALLOWED FILES TO MODIFY:
- task_agent.py (prompt, logic)
- hyper_agent.py (this file - you can modify your own prompt/logic)
- New helper modules (task agent can import them)

PROTECTED FILES (DO NOT MODIFY):
- config.py, domains/, agent/, utils/, generate_loop.py, harness.py, report.py

PATH STRUCTURE:
- Code: /hada/
- Previous gen results: {eval_path}/gen_N/ (e.g., gen_1/ for first generation)
- Your output: /hada/agent_output/

WHAT TO ANALYZE (focus on the latest generation):
- {eval_path}/gen_N/generate.log - Look for errors, timeouts, "[TRAIN] Run X failed:"
- {eval_path}/gen_N/<domain>_eval/agent_evals/task_agent_patch.diff - See what Task Agent changed
- {eval_path}/gen_N/<domain>_eval/agent_evals/chat_history_task_agent.md - See Task Agent's reasoning
- {eval_path}/gen_N/report.json - Evaluation scores

CRITICAL REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY: 

1. **YOU MUST MODIFY CODE FILES** - Use the `editor` tool with command='str_replace' to modify files
2. **DO NOT JUST VIEW FILES** - You must make ACTUAL CODE CHANGES using str_replace
3. **MANDATORY MODIFICATIONS** - You MUST modify at least one of these files:
   - /hada/task_agent.py (improve the prompt so Task Agent actually modifies EC code)
   - /hada/hyper_agent.py (improve your own prompt/logic)

4. **REQUIRED STEP-BY-STEP PROCESS**:
   Step 1: Use editor view command to see the current code/logs
   Step 2: Use editor str_replace command to make improvements (YOU MUST DO THIS)
   Step 3: Verify your changes were applied
   Step 4: Respond with JSON

## IMPORTANT GUIDANCE FOR TASK AGENT:

⚠️ **CRITICAL: The Task Agent MUST try different evolutionary algorithms. This is the #1 priority.**

Analysis of previous experiments shows that Task Agents consistently ONLY make small PSO parameter tweaks (adjusting w, c1, c2, adding turbulence, changing initialization) and NEVER replace the algorithm entirely. This has caused scores to plateau for 20+ generations.

You MUST ensure the task_agent.py prompt encourages the Task Agent to:

1. **Try different evolutionary algorithms** - not just parameter tweaks:
   - For **bbob_unconstrained**: DE, CMA-ES, SHADE, JADE, GLPSO, ES, etc.
   - For **bbob_constrained**: C-DE, CMA-ES with constraints, SHADE with feasibility rules, etc.
   - For **metabox_mt**: MFEA, MFEA-II, MFEA-DE, MFDE, CMT-DE, etc.
   - For **metabox_mo**: NSGA-II, NSGA-III, MOEA/D, SPEA2, SMPSO, IBEA, etc.

2. **Try different meta-learning approaches** (not just DQN)
   - PPO, Bayesian Optimization, L2O, etc.

3. **Make meaningful structural changes** - not just parameter tuning:
   - Change the search operators (mutation, crossover, selection)
   - Change the population structure
   - Add new mechanisms (archive, migration, restart)

4. **Actually modify the code** - not just describe changes

5. **INCREMENTAL IMPROVEMENT STRATEGY** - VERY IMPORTANT:
   - **Each generation should focus on ONE functional module at a time** (e.g., only change the evolutionary algorithm OR only change the meta-learning approach OR only change the population structure)
   - **Subsequent generations should gradually stack successful modules** (e.g., gen2 changes EA, gen3 changes meta-learning)
   - **Do NOT change everything at once** - this makes it impossible to identify what works
   - **Every generation MUST have a useful, actual code change** - no empty modifications

6. **EXPLORATION DIVERSITY** - CRITICAL:
   - **Do NOT limit to the current DE + DQN combination**
   - **Try completely different algorithm combinations**: CMA-ES + PPO, SHADE + Bayesian Opt, NSGA-III + L2O, etc.
   - **Explore different search operators, selection mechanisms, and parameter adaptation strategies**

7. **CHECK DOMAIN EVALUATION PATTERNS** - CRITICAL:
   - **ALWAYS check the domain folder (e.g., /hada/domains/bbob_unconstrained/dqn_de_util.py) to see how evaluation is called**
   - **The evaluation function is typically a module-level function like `evaluate_problem(problem, x)` - DO NOT convert it to a class method like `self._eval_task()` unless you define it first**
   - **Look at existing working code in the domain folder to understand the correct calling pattern before making changes**
   - **If you introduce a new method, you MUST define it in the same file before calling it**

8. **MAKE ACTUAL CODE IMPROVEMENTS AT EVERY STEP** - MANDATORY:
   - **Every generation MUST produce working code that can run successfully**
   - **Before submitting changes, verify: (a) all method calls exist, (b) function signatures match their callers, (c) no syntax errors**
   - **Do NOT introduce methods that don't exist - if you want to add a helper method, define it FIRST**
   - **Check that your modifications don't break existing functionality**

9. **CODE VERIFICATION CHECKLIST** - Before finishing:
   - [ ] All method/function calls reference existing code
   - [ ] No undefined variables or methods
   - [ ] Function signatures match their callers
   - [ ] The code can actually run without AttributeError or NameError
   - [ ] Changes are incremental and don't break existing functionality

Your goal is to ensure the Task Agent's prompt clearly communicates that it should try different algorithms, and that the Task Agent actually follows through with algorithm changes rather than just parameter tweaks.

GOAL: Improve the task_agent.py prompt/logic so Task Agent makes better modifications to the evolutionary algorithm and meta-learning code.
"""

        new_msg_history = chat_with_agent(
            instruction, 
            model=self.model, 
            msg_history=[], 
            logging=self.log, 
            tools_available='all', 
            require_tool='str_replace',
            multiple_tool_calls=True,
            max_tool_calls=100,
        )