"""
Multi-seed evaluation for DQN-DE across all domains.
Trains on seed 42 only, then tests the trained model on seeds 42-46.
"""
import os
import sys
import json
import csv
import numpy as np
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import set_global_seed


def run_multi_seed_eval(domain, output_dir, prev_gen_info=None):
    """Run DQN-DE evaluation on multiple seeds and produce report.
    
    Args:
        domain: one of 'bbob_unconstrained', 'bbob_constrained', 'metabox_mo'
        output_dir: directory to save results
        prev_gen_info: optional previous generation info
    
    Returns:
        report dict with mean_score, std_score, per_problem stats
    """
    # Import domain-specific modules
    if domain == 'bbob_unconstrained':
        from domains.bbob_unconstrained.dqn_de_util import run_de_and_score
        from domains.bbob_unconstrained.utils import format_input_dict
    elif domain == 'bbob_constrained':
        from domains.bbob_constrained.dqn_de_util import run_de_and_score
        from domains.bbob_constrained.utils import format_input_dict
    elif domain == 'metabox_mo':
        from domains.metabox_mo.dqn_de_util import run_de_and_score
        from domains.metabox_mo.utils import format_input_dict
    else:
        raise ValueError(f"Unsupported domain: {domain}")
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), domain, 'dataset.csv')
    dataset = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    
    train_dataset = [row for row in dataset if row.get('split') == 'train']
    test_dataset = [row for row in dataset if row.get('split') == 'test']
    
    train_seed = 42
    test_seeds = list(range(42, 47))
    num_sweeps = 5
    
    print(f"\n{'='*60}")
    print(f"Multi-seed evaluation for {domain}")
    print(f"Training seed: {train_seed}")
    print(f"Test seeds: {test_seeds}")
    print(f"Train problems: {len(train_dataset)}, Test problems: {len(test_dataset)}")
    print(f"{'='*60}")
    
    # Import meta_learning module
    this_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'metabbo')
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    spec2 = importlib.util.spec_from_file_location('meta_mod', os.path.join(this_dir, 'meta_learning.py'))
    meta_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(meta_mod)
    
    # ========== Phase 1: Train on seed 42 only ==========
    print(f"\n{'='*40}")
    print(f"=== Phase 1: Training on seed {train_seed} ===")
    print(f"{'='*40}")
    
    set_global_seed(train_seed)
    np.random.seed(train_seed)
    
    controller = meta_mod.DiscreteDQNController(training=True, seed=train_seed)
    
    sweep_returns = []
    step_losses = []
    
    for sweep in range(num_sweeps):
        sweep_return = 0.0
        n_problems = 0
        
        if hasattr(controller, 'reset_loss'):
            controller.reset_loss()
        
        for row in train_dataset:
            try:
                inputs = format_input_dict(row, prev_gen_info=prev_gen_info)
                result = run_de_and_score(inputs, log_fn=print, mode='train', controller=controller, seed=train_seed)
                
                if isinstance(result, dict):
                    score = result.get('score', 0.0)
                    sweep_return += score
                    n_problems += 1
                    if result.get('controller') is not None:
                        controller = result['controller']
            except Exception as e:
                print(f"  ERROR on problem {row.get('problem_id')}: {e}")
                continue
        
        sweep_loss = controller.get_total_loss() if hasattr(controller, 'get_total_loss') else 0.0
        sweep_returns.append(sweep_return)
        print(f"  Sweep {sweep+1}/{num_sweeps}: return={sweep_return:.4f}, loss={sweep_loss:.4f} ({n_problems} probs)")
    
    if hasattr(controller, 'get_loss_history'):
        step_losses = controller.get_loss_history()
    
    # Switch to test mode
    controller.training = False
    
    # ========== Phase 2: Test on all 5 seeds ==========
    print(f"\n{'='*40}")
    print(f"=== Phase 2: Testing on seeds {test_seeds} ===")
    print(f"{'='*40}")
    
    all_seed_scores = []
    all_seed_per_problem = {}
    all_seed_predictions = []
    
    for seed in test_seeds:
        print(f"\n--- Testing with seed {seed} ---")
        
        set_global_seed(seed)
        np.random.seed(seed)
        
        seed_scores = []
        seed_predictions = []
        
        for row in test_dataset:
            try:
                inputs = format_input_dict(row, prev_gen_info=prev_gen_info)
                result = run_de_and_score(inputs, log_fn=print, mode='test', controller=controller, seed=seed)
                
                if isinstance(result, dict):
                    score = result.get('score', 0.0)
                    seed_scores.append(score)
                    
                    # Build prediction string
                    if domain == 'metabox_mo':
                        pred_str = f"{result.get('score', 0.0)}|{result.get('hypervolume', 0.0)}|{result.get('igd', float('inf'))}|{result.get('gd', float('inf'))}|{result.get('spacing', 0.0)}|{result.get('n_solutions', 0)}"
                    else:
                        pred_str = f"{result.get('score', 0.0)}|{result.get('gbest_f', 0.0)}|{result.get('initial_gbest_f', 0.0)}|{result.get('opt_val', 0.0)}"
                    seed_predictions.append(pred_str)
                    
                    pid = row.get('problem_id', 'unknown')
                    if pid not in all_seed_per_problem:
                        all_seed_per_problem[pid] = []
                    all_seed_per_problem[pid].append(score)
                else:
                    seed_scores.append(0.0)
                    seed_predictions.append(str(result))
            except Exception as e:
                print(f"  ERROR on problem {row.get('problem_id')}: {e}")
                seed_scores.append(0.0)
                seed_predictions.append(f"ERROR: {e}")
        
        seed_mean = np.mean(seed_scores)
        seed_std = np.std(seed_scores)
        all_seed_scores.append(seed_mean)
        all_seed_predictions.append(seed_predictions)
        
        print(f"[Seed {seed}] Mean Score: {seed_mean:.4f} +/- {seed_std:.4f}")
    
    # Overall results
    overall_mean = np.mean(all_seed_scores)
    overall_std = np.std(all_seed_scores)
    
    print(f"\n{'='*60}")
    print(f"Overall Results for {domain}")
    print(f"{'='*60}")
    print(f"Mean Score: {overall_mean:.4f} +/- {overall_std:.4f}")
    print(f"Per-seed means: {[f'{s:.4f}' for s in all_seed_scores]}")
    
    # Per-problem stats
    per_problem_stats = {}
    for pid, scores in all_seed_per_problem.items():
        per_problem_stats[pid] = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'scores': scores
        }
        print(f"  {pid}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    
    # Save report
    report = {
        'score': float(overall_mean),
        'overall_mean_score': float(overall_mean),
        'overall_std_score': float(overall_std),
        'per_seed_means': [float(s) for s in all_seed_scores],
        'per_problem': per_problem_stats,
        'train_seed': train_seed,
        'test_seeds': test_seeds,
        'sweep_returns': sweep_returns,
        'step_losses': step_losses,
    }
    
    report_path = os.path.join(output_dir, 'report.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {report_path}")
    
    # Save predictions.csv (average across seeds)
    avg_predictions = []
    for idx in range(len(test_dataset)):
        preds_for_problem = []
        for seed_preds in all_seed_predictions:
            if idx < len(seed_preds):
                pred_str = seed_preds[idx]
                parts = str(pred_str).split('|')
                if len(parts) >= 1:
                    try:
                        preds_for_problem.append(float(parts[0]))
                    except ValueError:
                        preds_for_problem.append(0.0)
        
        avg_score = np.mean(preds_for_problem) if preds_for_problem else 0.0
        
        # Use the first seed's prediction format but with averaged score
        if all_seed_predictions and len(all_seed_predictions[0]) > idx:
            first_pred = all_seed_predictions[0][idx]
            parts = str(first_pred).split('|')
            if len(parts) >= 4:
                avg_pred = f"{avg_score}|{parts[1]}|{parts[2]}|{parts[3]}"
            else:
                avg_pred = f"{avg_score}"
        else:
            avg_pred = f"{avg_score}"
        
        avg_predictions.append(avg_pred)
    
    # Write predictions.csv
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    with open(predictions_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['problem_id', 'function', 'instance', 'dimension', 'opt_val', 'split', 'prediction'])
        for row, pred in zip(test_dataset, avg_predictions):
            writer.writerow([
                row.get('problem_id', ''),
                row.get('function', ''),
                row.get('instance', ''),
                row.get('dimension', ''),
                row.get('opt_val', ''),
                'test',
                pred
            ])
    
    print(f"Predictions saved to {predictions_path}")
    
    # Generate training curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Figure 1: Return curve
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.plot(range(1, len(sweep_returns) + 1), sweep_returns, 'b-o', linewidth=2.5, markersize=8, alpha=0.8)
        ax1.set_xlabel('Sweep', fontsize=12)
        ax1.set_ylabel('Sweep Return', fontsize=12)
        ax1.set_title(f'{domain} - Sweep Return (Training on seed {train_seed})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path1 = os.path.join(output_dir, f'{domain}_return_curves.png')
        plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Return curve saved to {plot_path1}")
        
        # Figure 2: Loss curve
        if step_losses:
            downsampled_losses = step_losses[::5]
            losses_to_plot = downsampled_losses
            
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            use_log_scale = False
            if max(losses_to_plot) > 100 * min([x for x in losses_to_plot if x > 0] + [1.0]):
                losses_to_plot = [np.log1p(x) for x in losses_to_plot]
                use_log_scale = True
            
            ax2.plot(range(1, len(losses_to_plot) + 1), losses_to_plot, 'r-', linewidth=1.5, alpha=0.7)
            ax2.set_xlabel('Update Interval (every 10 steps)', fontsize=12)
            ax2.set_ylabel('Training Loss (log scale)' if use_log_scale else 'Training Loss', fontsize=12)
            ax2.set_title(f'{domain} - DQN Training Loss per 10 Updates', fontsize=14)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path2 = os.path.join(output_dir, f'{domain}_loss_curves.png')
            plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Loss curve saved to {plot_path2}")
        
    except Exception as e:
        print(f"Warning: Failed to generate training graphs: {e}")
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, 
                       choices=['bbob_unconstrained', 'bbob_constrained', 'metabox_mo'])
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    run_multi_seed_eval(args.domain, args.output_dir)