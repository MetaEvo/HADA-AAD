import os
import sys

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import global configuration
sys.path.insert(0, '/hada')
from config import set_global_seed, SEED, MODEL_NAME
from utils.constants import REPO_NAME

# Set global seed at module load time
set_global_seed()

import argparse
import importlib
import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
from hydra import compose, initialize_config_dir
from types import ModuleType


def get_dataset(domain, subset=""):
    df = None
    if "imo_" in domain:
        df = pd.read_csv(f"./domains/imo/{domain.split('_')[-1]}bench{subset}.csv", dtype=str)
    elif domain in ["search_arena", "paper_review", "bbob_unconstrained", "bbob_constrained", "metabox_mo", "metabox_mt"]:
        df = pd.read_csv(f"./domains/{domain}/dataset{subset}.csv", dtype=str)
    return df

def run_agent(TaskAgent, model, row, evals_folder, format_input_dict, question_id_col, run_task_agent=True, prev_gen_info=None, generate_log_path=None):
    question_id = row[question_id_col]
    chat_history_path = os.path.join(evals_folder, f"chat_history_{question_id}.md")
    agent = TaskAgent(model=model, chat_history_file=chat_history_path)
    
    # 如果提供了 generate_log_path，设置日志输出
    if generate_log_path:
        import logging
        log_handler = logging.FileHandler(generate_log_path)
        log_handler.setLevel(logging.INFO)
        log_formatter = logging.Formatter("%(asctime)s - %(message)s")
        log_handler.setFormatter(log_formatter)
        thread_logger = logging.getLogger(f"run_agent_{question_id}")
        thread_logger.setLevel(logging.INFO)
        thread_logger.addHandler(log_handler)
        
        def log_with_file(msg):
            msg_str = str(msg)
            thread_logger.info(msg_str)
            print(msg_str, flush=True)
        
        agent.log = log_with_file
    
    inputs = format_input_dict(row, prev_gen_info=prev_gen_info)
    
    # 对于 bbob_unconstrained, bbob_constrained, metabox_mo, metabox_mt domain，只调用 dqn_pso_util 运行 PSO
    domain = inputs.get('domain')
    if domain in ['bbob_unconstrained', 'bbob_constrained', 'metabox_mo', 'metabox_mt']:
        if domain == 'bbob_unconstrained':
            from domains.bbob_unconstrained.dqn_pso_util import run_pso_and_score
        elif domain == 'bbob_constrained':
            from domains.bbob_constrained.dqn_pso_util import run_pso_and_score
        elif domain == 'metabox_mo':
            from domains.metabox_mo.dqn_pso_util import run_pso_and_score
        else:  # metabox_mt
            from domains.metabox_mt.dqn_pso_util import run_pso_and_score
        
        try:
            result = run_pso_and_score(inputs, log_fn=agent.log)
            
            # 如果返回的是字典，转换为字符串格式
            if isinstance(result, dict):
                if domain == 'metabox_mo':
                    # metabox_mo 格式: score|hypervolume|igd|n_solutions
                    score = result.get('score', 0.0)
                    hv = result.get('hypervolume', 0.0)
                    igd = result.get('igd', float('inf'))
                    n_sol = result.get('n_solutions', 0)
                    return f"{score}|{hv}|{igd}|{n_sol}"
                elif domain == 'metabox_mt':
                    # metabox_mt 格式: score|avg_task_score|knowledge_transfer|n_tasks|task_scores|task_gbest_f|task_similarity
                    score = result.get('score', 0.0)
                    avg_task = result.get('avg_task_score', 0.0)
                    kte = result.get('knowledge_transfer', 0.0)
                    n_tasks = result.get('n_tasks', 0)
                    task_scores = json.dumps(result.get('task_scores', []))
                    task_gbest_f = json.dumps(result.get('task_gbest_f', []))
                    task_similarity = result.get('task_similarity', 0.5)
                    return f"{score}|{avg_task}|{kte}|{n_tasks}|{task_scores}|{task_gbest_f}|{task_similarity}"
            
            # bbob 返回的是字符串，直接返回
            return result
        except Exception as e:
            error_msg = f"[EVAL] Problem {question_id} failed: {e}"
            agent.log(error_msg)
            import traceback
            agent.log(f"Traceback: {traceback.format_exc()}")
            return f"ERROR: {e}"
    
    # 其他 domain 直接调用 task_agent
    prediction, _ = agent.forward(inputs)
    return prediction


def load_task_agent(agent_path: str):
    """
    agent_path can be:
      - a python file path: ./task_agent.py or /abs/path/task_agent.py
      - a module path: proofgrader.task_agent or my_pkg.my_agent
    Returns: TaskAgent class
    """
    # Case 1: looks like a file path or exists on disk
    if agent_path.endswith(".py") or os.path.exists(agent_path):
        abs_path = os.path.abspath(agent_path)
        spec = importlib.util.spec_from_file_location("agent_module", abs_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from file: {abs_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "TaskAgent"):
            raise AttributeError(f"No TaskAgent found in file: {abs_path}")
        return mod.TaskAgent

    # Case 2: interpret as module path
    mod = importlib.import_module(agent_path)
    if not hasattr(mod, "TaskAgent"):
        raise AttributeError(f"No TaskAgent found in module: {agent_path}")
    return mod.TaskAgent

def harness(
    agent_path="./task_agent.py",
    output_dir="./outputs",
    run_id=None,
    domain="search_arena",
    num_samples=-1,
    save_interval=100,
    num_workers=5,
    resume_from=None,
    subset="",
    proofs_dname=None,
    prev_gen_info=None,
):
    # Dynamically import functions based on the domain
    utils_prefix = domain.split("_", 1)[1] + "_" if domain.startswith("imo_") else ""
    # 对于 metabox_mo 和 metabox_mt，使用完整的 domain 名称作为文件夹
    if domain.startswith("metabox_"):
        domain_folder = domain
    else:
        domain_folder = domain.split('_')[0] if "imo_" in domain else domain
    utils_module_path = f"domains.{domain_folder}.{utils_prefix}utils"
    utils_module = importlib.import_module(utils_module_path)
    format_input_dict = utils_module.format_input_dict
    question_id_col = utils_module.QUESTION_ID
    model = utils_module.MODEL

    # Load TaskAgent either from a file path or an importable module path
    TaskAgent = load_task_agent(agent_path)

    # Specify output folder
    if resume_from:
        output_folder = os.path.abspath(resume_from)
    else:
        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") if run_id is None else run_id
        )
        output_folder = os.path.join(os.getcwd(), output_dir, run_id)

    # Create output folder
    evals_folder = os.path.join(output_folder, "agent_evals")
    os.makedirs(evals_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "predictions.csv")
    
    # 设置 generate.log 文件路径（用于记录 task agent 和评估日志）
    generate_log_path = os.path.join(output_folder, "generate.log")

    # Load existing predictions if available
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path, dtype=str)
        completed_ids = set(
            existing_df[~existing_df["prediction"].isna()][question_id_col]
        )
    else:
        existing_df = None
        completed_ids = set()

    # Get dataset
    if proofs_dname:
        dataset = pd.read_csv(os.path.join(proofs_dname, "predictions.csv"), dtype=str)
        dataset["Response"] = dataset["prediction"].copy()
        dataset.drop(columns=["prediction"], inplace=True)
    else:
        dataset = get_dataset(domain=domain, subset=subset)
    if num_samples > 0:
        dataset = dataset[:num_samples]

    # 对于 bbob_unconstrained, bbob_constrained, metabox_mo, metabox_mt domain，先调用 task_agent 一次修改代码，然后训练 DQN，最后测试
    if domain in ['bbob_unconstrained', 'bbob_constrained', 'metabox_mo', 'metabox_mt']:
        chat_history_path = os.path.join(evals_folder, "chat_history_task_agent.md")
        
        # 设置日志输出到 generate.log
        import logging
        import threading
        
        # 创建 generate.log 的文件处理器
        log_handler = logging.FileHandler(generate_log_path)
        log_handler.setLevel(logging.INFO)
        log_formatter = logging.Formatter("%(asctime)s - %(message)s")
        log_handler.setFormatter(log_formatter)
        
        # 创建一个专用的 logger
        generate_logger = logging.getLogger(f"generate_{domain}")
        generate_logger.setLevel(logging.INFO)
        generate_logger.addHandler(log_handler)
        
        # 定义日志回调函数，同时输出到 generate.log 和打印到控制台
        def log_to_generate(msg):
            msg_str = str(msg)
            generate_logger.info(msg_str)
            print(msg_str, flush=True)
        
        agent = TaskAgent(model=model, chat_history_file=chat_history_path)
        # 覆盖 agent.log 方法，使其输出到 generate.log
        agent.log = log_to_generate
        
        # 使用第一个 problem 的 input 调用 task_agent
        sample_row = dataset.iloc[0].to_dict()
        sample_inputs = format_input_dict(sample_row, prev_gen_info=prev_gen_info)
        
        # 直接在 harness.py 中构建完整的 instruction，像 gen_initial 版本那样
        from agent.llm_withtools import chat_with_agent
        from utils.common import extract_jsons
        
        # 获取项目根目录的相对路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cocoex_path = os.path.join(project_root, 'metabbo')
        
        full_instruction = f"""You are an expert agent specialized in solving tasks within the {sample_inputs['domain']} domain.

Task input:
```
{sample_inputs}
```

⚠️⚠️⚠️ CRITICAL REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY: ⚠️⚠️⚠️

1. **YOU MUST MODIFY CODE FILES** - Use the `editor` tool with command='str_replace' to modify files in `{cocoex_path}/`
2. **DO NOT JUST VIEW FILES** - You must make ACTUAL CODE CHANGES using str_replace
3. **MANDATORY MODIFICATIONS** - You MUST modify at least one of these files:
   - {cocoex_path}/pso.py
   - {cocoex_path}/dqn_controller.py  
   - {cocoex_path}/param_controller.py

4. **REQUIRED STEP-BY-STEP PROCESS**:
   Step 1: Use editor view command to see the current code
   Step 2: Use editor str_replace command to make improvements (YOU MUST DO THIS)
   Step 3: Verify your changes were applied
   Step 4: Respond with JSON

5. **EXAMPLE OF str_replace USAGE**:
   <json>
   {{
       "tool_name": "editor",
       "tool_input": {{
           "command": "str_replace",
           "path": "{cocoex_path}/pso.py",
           "old_str": "def optimize(self):\\n    # old code here",
           "new_str": "def optimize(self):\\n    # improved code here"
       }}
   }}
   </json>

6. **CODE CORRECTNESS REQUIREMENTS** (CRITICAL - PREVENTS ZERO SCORE):
   - **CHECK VARIABLE SCOPE**: Ensure all variables you use are defined in the current scope
   - **VERIFY INDENTATION**: Python is indentation-sensitive - maintain correct nesting
   - **INITIALIZE VARIABLES**: If you use new variables (like 'archive', 'f', etc.), initialize them first
   - **TEST LOGIC FLOW**: Your code should not break the existing PSO loop structure
   - **AVOID UNDEFINED REFERENCES**: Do NOT reference variables from inner loops in outer scopes

7. **ZERO SCORE WARNINGS**:
   - If you do NOT call str_replace at least once → ZERO score
   - If your code has syntax errors or runtime errors → ZERO score
   - If your code references undefined variables → ZERO score

Your goal is to improve the PSO algorithm's performance on the {domain} problems.

After completing all modifications, you MUST respond in JSON format:
<json>
{{
    "response": "Brief description of the changes made"
}}
</json>"""
        
        agent.log(f"Calling task agent once to modify code for {domain} domain...")
        agent.log(f"Task instruction length: {len(full_instruction)} chars")
        agent.log(f"Enabling multiple tool calls for faster execution...")
        agent.log(f"Model: {model}")
        
        try:
            new_msg_history = chat_with_agent(
                full_instruction, 
                model=model, 
                msg_history=[], 
                logging=agent.log, 
                tools_available='all',
                multiple_tool_calls=True,  # 启用多工具调用，减少往返次数
                require_tool='str_replace'  # 强制要求调用 str_replace
            )
            agent.log(f"Task agent chat completed, {len(new_msg_history)} messages")
        except Exception as e:
            agent.log(f"ERROR in chat_with_agent: {e}")
            import traceback
            agent.log(traceback.format_exc())
            new_msg_history = [{"role": "assistant", "text": "Error during task agent execution"}]
        
        # 提取 prediction
        prediction = "None"
        try:
            extracted_jsons = extract_jsons(new_msg_history[-1]['text'])
            if extracted_jsons is not None and "response" in extracted_jsons[-1]:
                prediction = extracted_jsons[-1]["response"]
        except Exception as e:
            agent.log(f"Error extracting prediction: {e}")
            prediction = "None"
        
        # 保存 chat history
        import json
        with open(chat_history_path, 'w') as f:
            json.dump(new_msg_history, f, indent=2)
        
        # 生成 all_patch.diff
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'diff', '--no-color'],
                capture_output=True,
                text=True,
                cwd='/hada'
            )
            all_patch_after_task = result.stdout
            patch_path = os.path.join(evals_folder, "all_patch.diff")
            with open(patch_path, 'w') as f:
                f.write(all_patch_after_task)
            if all_patch_after_task:
                agent.log(f"Saved all patch to {patch_path}")
            else:
                agent.log("No changes detected by git diff, saved empty patch")
        except Exception as e:
            agent.log(f"Error generating patch: {e}")
        
        agent.log(f"Task agent completed. Prediction: {prediction}")
        
        # 分离训练集和测试集
        agent.log(f"Dataset columns: {dataset.columns.tolist()}")
        agent.log(f"Dataset shape: {dataset.shape}")
        if 'split' in dataset.columns:
            agent.log(f"Split values: {dataset['split'].unique()}")
            train_dataset = dataset[dataset['split'] == 'train'].copy()
            test_dataset = dataset[dataset['split'] == 'test'].copy()
            agent.log(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
        else:
            # 如果没有 split 列，使用前 8 个作为训练，其余作为测试
            agent.log("No split column found, using first 8 as train, rest as test")
            train_dataset = dataset.iloc[:8].copy()
            test_dataset = dataset.iloc[8:].copy()
        
        # 根据 domain 导入正确的 dqn_pso_util
        if domain == 'bbob_unconstrained':
            from domains.bbob_unconstrained.dqn_pso_util import run_pso_and_score
        elif domain == 'bbob_constrained':
            from domains.bbob_constrained.dqn_pso_util import run_pso_and_score
        elif domain == 'metabox_mo':
            from domains.metabox_mo.dqn_pso_util import run_pso_and_score
        elif domain == 'metabox_mt':
            from domains.metabox_mt.dqn_pso_util import run_pso_and_score
        
        # 步骤 1：用训练集训练 DQN 模型（多个 sweep，共用同一个 DQN）
        agent.log(f"\n=== Training DQN on {len(train_dataset)} training functions ===")
        num_sweeps = 10  # 跑 10 个 sweep
        
        # 收集训练数据用于画图
        sweep_returns = []  # 每个 sweep 的总 return
        training_losses = []  # 每个 problem 的 training loss
        # 找到 gen_N 目录
        gen_dir = output_folder
        while gen_dir and not os.path.basename(gen_dir).startswith('gen_'):
            parent = os.path.dirname(gen_dir)
            if parent == gen_dir:
                break
            gen_dir = parent
        if not os.path.basename(gen_dir).startswith('gen_'):
            gen_dir = output_folder
        
        for sweep in range(num_sweeps):
            agent.log(f"\n--- Sweep {sweep + 1}/{num_sweeps} ---")
            n_problems = 0
            sweep_losses = []  # 当前 sweep 所有 problem 的 loss
            
            for _, row in train_dataset.iterrows():
                try:
                    inputs = format_input_dict(row, prev_gen_info=prev_gen_info)
                    # 训练模式：run_pso_and_score 会更新 DQN 模型
                    result = run_pso_and_score(inputs, log_fn=agent.log, mode='train')
                    
                    # 收集训练数据
                    if isinstance(result, dict):
                        if result.get('training_loss') is not None:
                            sweep_losses.append(result['training_loss'])
                        n_problems += 1
                except Exception as e:
                    error_msg = f"[TRAIN] Run {sweep + 1} failed: {e}"
                    agent.log(error_msg)
                    import traceback
                    agent.log(f"Traceback: {traceback.format_exc()}")
                    continue
            
            # Get DQN return from controller (entire sweep's cumulative reward)
            sweep_return = shared_controller.get_total_reward() if hasattr(shared_controller, 'get_total_reward') else 0.0
            sweep_returns.append(sweep_return)
            # 计算当前 sweep 的平均 loss
            avg_loss = sum(sweep_losses) / len(sweep_losses) if sweep_losses else None
            training_losses.append(avg_loss)
            agent.log(f"[{domain}] Sweep {sweep + 1} total return: {sweep_return:.4f} (from {n_problems} problems)")
            if avg_loss is not None:
                agent.log(f"[{domain}] Sweep {sweep + 1} avg training loss: {avg_loss:.4f}")
        
        # 每个 gen 训练结束后：记录数组 + 画图
        agent.log(f"\n[{domain}] DQN Training Loss Array: {training_losses}")
        agent.log(f"[{domain}] DQN Sweep Return Array: {sweep_returns}")
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import networkx as nx
            from networkx.drawing.nx_agraph import graphviz_layout
            
            # 左图：折线图
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(range(1, len(sweep_returns) + 1), sweep_returns, 'b-o', linewidth=2, markersize=8)
            axes[0].set_xlabel('Sweep', fontsize=12)
            axes[0].set_ylabel('Total Return', fontsize=12)
            axes[0].set_title(f'{domain} - Sweep Total Return', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xticks(range(1, len(sweep_returns) + 1))
            
            if training_losses:
                axes[1].plot(range(1, len(training_losses) + 1), training_losses, 'r-o', linewidth=2, markersize=6)
                axes[1].set_xlabel('Sweep', fontsize=12)
                axes[1].set_ylabel('Avg Training Loss', fontsize=12)
                axes[1].set_title(f'{domain} - DQN Avg Training Loss per Sweep', fontsize=14)
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(gen_dir, f'{domain}_training_curves.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            agent.log(f"[{domain}] Training curves saved to {plot_path}")
            
            # 右图：networkx + graphviz 进化图（loss 和 return）
            G = nx.DiGraph()
            score_map = {}
            for i in range(len(sweep_returns)):
                G.add_node(str(i))
                if i > 0:
                    G.add_edge(str(i - 1), str(i))
                score_map[str(i)] = sweep_returns[i]
            
            if len(G.nodes()) > 0:
                pos = graphviz_layout(G, prog='dot')
                
                nodes = list(G.nodes())
                scores = [score_map.get(node, None) for node in nodes]
                valid_scores = [s for s in scores if s is not None]
                
                if len(valid_scores) == 0:
                    min_score, max_score = 0.0, 1.0
                else:
                    min_score, max_score = min(valid_scores), max(valid_scores)
                
                normalized_scores = []
                for s in scores:
                    if s is None:
                        normalized_scores.append(0.0)
                    else:
                        normalized_scores.append((s - min_score) / (max_score - min_score) if max_score != min_score else 0.5)
                
                from matplotlib.colors import LinearSegmentedColormap
                colors = [(0.0, "#ffaf00"), (0.4, "#ffd000"), (0.8, "#fffb00"), (1.0, "#03ff00")]
                cmap = LinearSegmentedColormap.from_list("orange_yellow_green", colors, N=256)
                node_colors = cmap(normalized_scores)
                
                valid_score_map = {k: v for k, v in score_map.items() if v is not None}
                max_score_node = max(valid_score_map, key=valid_score_map.get) if valid_score_map else None
                
                fig, ax = plt.subplots(figsize=(16, 10))
                for i, n in enumerate(nodes):
                    shape = 'D' if (max_score_node is not None and n == max_score_node) else 'o'
                    nx.draw_networkx_nodes(G, pos, nodelist=[n], node_color=[node_colors[i]], edgecolors=["black"], node_size=800, node_shape=shape, linewidths=1, ax=ax)
                
                nx.draw_networkx_edges(G, pos, ax=ax, arrows=False)
                labels = {}
                for n in nodes:
                    s = score_map.get(n)
                    if s is None:
                        labels[n] = f"Sweep {int(n)+1}\nN/A"
                    else:
                        labels[n] = f"Sweep {int(n)+1}\n{s:.2f}"
                nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
                ax.axis('off')
                plt.tight_layout()
                
                graph_path = os.path.join(gen_dir, f'{domain}_return_graph.png')
                plt.savefig(graph_path, dpi=150, bbox_inches='tight')
                plt.close()
                agent.log(f"[{domain}] Return graph saved to {graph_path}")
            
            # Loss graph (networkx style)
            if training_losses:
                G2 = nx.DiGraph()
                loss_map = {}
                for i in range(len(training_losses)):
                    G2.add_node(str(i))
                    if i > 0:
                        G2.add_edge(str(i - 1), str(i))
                    loss_map[str(i)] = training_losses[i]
                
                pos2 = {str(i): (i, training_losses[i]) for i in range(len(training_losses))}
                nodes2 = list(G2.nodes())
                losses = [loss_map.get(node, None) for node in nodes2]
                
                valid_loss_map = {k: v for k, v in loss_map.items() if v is not None}
                min_loss_node = min(valid_loss_map, key=valid_loss_map.get) if valid_loss_map else None
                
                # Check if we need log scale (max/min ratio > 100)
                valid_loss_values = [v for v in loss_map.values() if v is not None and v > 0]
                use_log = False
                if len(valid_loss_values) >= 2:
                    ratio = max(valid_loss_values) / min(valid_loss_values)
                    if ratio > 100:
                        use_log = True
                
                fig2, ax2 = plt.subplots(figsize=(16, 10))
                for i, n in enumerate(nodes2):
                    shape = 'D' if (min_loss_node is not None and n == min_loss_node) else 'o'
                    nx.draw_networkx_nodes(G2, pos2, nodelist=[n], node_color="red", edgecolors=["black"], node_size=800, node_shape=shape, linewidths=1, ax=ax2)
                
                nx.draw_networkx_edges(G2, pos2, ax=ax2, arrows=False, width=2)
                loss_labels = {}
                for n in nodes2:
                    v = loss_map.get(n)
                    if v is None:
                        loss_labels[n] = f"Sweep {int(n)+1}\nN/A"
                    else:
                        loss_labels[n] = f"Sweep {int(n)+1}\n{v:.2f}"
                nx.draw_networkx_labels(G2, pos2, labels=loss_labels, ax=ax2, font_size=10)
                
                ax2.set_xlabel('Sweep', fontsize=14)
                ax2.set_ylabel('Avg Training Loss', fontsize=14)
                ax2.set_title(f'{domain} - DQN Avg Training Loss per Sweep', fontsize=16)
                ax2.grid(True, alpha=0.3)
                ax2.set_xticks(range(len(training_losses)))
                ax2.set_xticklabels([f'Sweep {i+1}' for i in range(len(training_losses))])
                
                if use_log:
                    ax2.set_yscale('log')
                
                plt.tight_layout()
                
                loss_graph_path = os.path.join(gen_dir, f'{domain}_loss_graph.png')
                plt.savefig(loss_graph_path, dpi=150, bbox_inches='tight')
                plt.close()
                agent.log(f"[{domain}] Loss graph saved to {loss_graph_path}")
                
        except Exception as e:
            agent.log(f"Warning: Failed to generate training graphs: {e}")
        
        # 步骤 2：用测试集评估（不再训练，只使用训练后的 DQN）
        agent.log(f"\n=== Evaluating on {len(test_dataset)} test functions ===")
        dataset = test_dataset.copy().reset_index(drop=True)
        dataset["prediction"] = None
        predictions = [None] * len(test_dataset)
        
        # 在测试集上运行评估
        for idx, row in dataset.iterrows():
            try:
                inputs = format_input_dict(row, prev_gen_info=prev_gen_info)
                # 测试模式：只评估，不训练
                result = run_pso_and_score(inputs, log_fn=agent.log, mode='test')
                # 将结果转换为字符串格式存储
                if isinstance(result, dict):
                    if domain == 'metabox_mo':
                        prediction_str = f"{result.get('score', 0.0)}|{result.get('hypervolume', 0.0)}|{result.get('igd', float('inf'))}|{result.get('n_solutions', 0)}"
                    elif domain == 'metabox_mt':
                        # metabox_mt 格式: score|avg_task_score|knowledge_transfer|n_tasks
                        prediction_str = f"{result.get('score', 0.0)}|{result.get('avg_task_score', 0.0)}|{result.get('knowledge_transfer', 0.0)}|{result.get('n_tasks', 0)}"
                    elif domain == 'bbob_unconstrained':
                        # bbob_unconstrained 格式: score|gbest_f|initial_gbest_f|opt_val
                        prediction_str = f"{result.get('score', 0.0)}|{result.get('gbest_f', 0.0)}|{result.get('initial_gbest_f', 0.0)}|{result.get('opt_val', 0.0)}"
                    else:
                        prediction_str = str(result.get('score', 0.0))
                else:
                    prediction_str = str(result)
                predictions[idx] = prediction_str
                dataset.at[idx, "prediction"] = prediction_str
            except Exception as e:
                error_msg = f"[EVAL] Problem {idx} failed: {e}"
                agent.log(error_msg)
                import traceback
                agent.log(f"Traceback: {traceback.format_exc()}")
                predictions[idx] = f"ERROR: {e}"
                dataset.at[idx, "prediction"] = predictions[idx]
                continue
            
            if (idx + 1) % save_interval == 0:
                dataset.to_csv(output_path, index=False)
                agent.log(f"Checkpoint saved to {output_path}")
        
        # Final save for metabox domains
        dataset["prediction"] = predictions
        dataset.to_csv(output_path, index=False)
        agent.log(f"Final predictions saved to {output_path}")
        
        return output_folder
    else:
        # Add a prediction column
        if existing_df is not None:
            dataset = dataset.merge(
                existing_df[[question_id_col, "prediction"]], on=question_id_col, how="left"
            )
        else:
            dataset["prediction"] = None
        predictions = dataset["prediction"].tolist()
    futures = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, row in dataset.iterrows():
            if (
                pd.notna(row["prediction"]) or row[question_id_col] in completed_ids
            ):  # pyright: ignore
                continue
            futures.append(
                (
                    i,
                    executor.submit(
                        run_agent,
                        TaskAgent, model, row, evals_folder,
                        format_input_dict, question_id_col,
                        prev_gen_info=prev_gen_info,
                        generate_log_path=generate_log_path,
                    ),
                )
            )

        for idx, future in futures:
            prediction = future.result()
            predictions[idx] = prediction

            if (idx + 1) % save_interval == 0:
                dataset["prediction"] = predictions
                dataset.to_csv(output_path, index=False)
                print(f"Checkpoint saved to {output_path}")

    # Final save
    dataset["prediction"] = predictions
    dataset.to_csv(output_path, index=False)
    print(f"Final predictions saved to {output_path}")

    return output_folder


def main():
    """Main entry point for harness."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a system on the search arena dataset."
    )
    parser.add_argument(
        "--agent_path", type=str, default="./task_agent.py", help="Path to the agent"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--run_id", type=str, default=None, help="Run ID")
    parser.add_argument(
        "--domain",
        type=str,
        choices=[
            "search_arena",
            "paper_review",
            "balrog_babyai",
            "balrog_babaisai",
            "balrog_minihack",
            "balrog_nle",
            "genesis_go2walking",
            "genesis_go2walkback",
            "genesis_go2hop",
            "imo_grading",
            "imo_proof",
            "imo_proof_grading",  # To grade generated proofs with an agent
            "bbob_unconstrained",  # To evaluate PSO agent on COCOEX BBOB benchmark
            "bbob_constrained",  # To evaluate PSO agent on COCOEX BBOB constrained benchmark
            "metabox_mo",  # To evaluate PSO agent on MetaBox multi-objective benchmark
            "metabox_mt",  # To evaluate PSO agent on MetaBox multi-task benchmark
        ],
        required=True,
        help="Domain to evaluate",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate, -1 for all",
    )
    parser.add_argument(
        "--save_interval", type=int, default=100, help="Save to CSV every n samples"
    )
    parser.add_argument(
        "--num_workers", type=int, default=5, help="Number of parallel workers"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an existing output folder to resume from",
    )
    parser.add_argument(
        "--subset", type=str, default="", help="Subset of the dataset to evaluate"
    )
    parser.add_argument(
        "--proofs_dname", type=str, default="", help="Path to the directory containing proofs to grade (for imo_proof_grading)"
    )
    args = parser.parse_args()
    # 清理 sys.argv，防止底层的 metaevobox 等第三方库在初始化时误读取这些参数导致崩溃
    sys.argv = [sys.argv[0]]
    domain = args.domain
   # Make proofs_dname required for imo_proof_grading
    if domain == "imo_proof_grading" and not args.proofs_dname:
        parser.error("--proofs_dname is required when domain is 'imo_proof_grading'")

    # Human preferences domains (including bbob and metabox)
    if domain in ["search_arena", "paper_review", "imo_grading", "imo_proof", "imo_proof_grading", "bbob_unconstrained", "bbob_constrained", "metabox_mo", "metabox_mt"]:
        # 从容器内的文件读取 prev_gen_info（generate_loop.py 写入）
        prev_info = None
        prev_info_file = f"/{REPO_NAME}/prev_gen_info_{domain}.json"
        if os.path.exists(prev_info_file):
            try:
                import json as json_mod
                with open(prev_info_file, 'r') as f:
                    prev_info = json_mod.load(f)
                print(f"DEBUG: Loaded prev_gen_info from {prev_info_file}")
            except Exception as e:
                print(f"DEBUG: Error loading prev_gen_info: {e}")
        
        output_folder = harness(
            agent_path=args.agent_path,
            output_dir=args.output_dir,
            run_id=args.run_id,
            domain=args.domain,
            num_samples=args.num_samples,
            save_interval=args.save_interval,
            num_workers=args.num_workers,
            resume_from=args.resume_from,
            subset=args.subset,
            proofs_dname=args.proofs_dname,
            prev_gen_info=prev_info,
        )

    # Balrog game domains
    elif "balrog" in domain:
        from domains.balrog.eval import harness_balrog

        env_name = domain.split("_")[-1]
        config_dir = os.path.join(os.getcwd(), "./domains/balrog/config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"eval.output_dir={args.output_dir}",
                    f"eval.num_workers={args.num_workers}",
                    f"envs.names={env_name}",
                    f"eval.run_id={args.run_id if args.run_id is not None else 'null'}",
                ]
                + (
                    [f"eval.num_episodes.{env_name}={args.num_samples}"]
                    if args.num_samples > 0
                    else []
                )
                + (
                    [f"eval.resume_from={args.resume_from}"]
                    if args.resume_from is not None
                    else []
                ),
            )
            output_folder = harness_balrog(cfg)
            # Save cfg in output folder
            from omegaconf import OmegaConf
            OmegaConf.save(config=cfg, f=os.path.join(output_folder, "config.yaml"))

    # Genesis Robotic Control Domains
    elif "genesis" in domain:
        from domains.genesis.eval import harness_genesis

        env_name = domain.split("_")[-1]
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_dir = os.path.join(root_dir, "domains/genesis/config")
        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            num_workers = 1
            cfg = compose(
                config_name="config",
                overrides=[
                    f"eval.output_dir={args.output_dir}",
                    f"eval.num_workers={num_workers}",
                    f"envs.names={env_name}",
                    f"eval.run_id={args.run_id if args.run_id is not None else 'null'}",
                    f"utils.root_dir={root_dir}",
                ]
                + (
                    [f"eval.num_episodes.{env_name}={args.num_samples}"]
                    if args.num_samples > 0
                    else []
                )
            )
            output_folder = harness_genesis(cfg)
            # Save cfg in output folder
            from omegaconf import OmegaConf
            OmegaConf.save(config=cfg, f=os.path.join(output_folder, "config.yaml"))


if __name__ == "__main__":
    main()