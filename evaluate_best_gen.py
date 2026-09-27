"""
Evaluate the best generation algorithm from HADA run.
Runs on test problems with seeds 42-46, computes RI scores.
"""
import sys
import os
import csv
import json
import numpy as np
import time

sys.path.insert(0, '/hada')

# Import the best gen algorithm
BEST_GEN_PATH = '/hada/outputs/成功运行/generate_20260901_221708_346043/metabbo_best_gen'
sys.path.insert(0, BEST_GEN_PATH)

from ec_algorithm import DEOptimizer
from meta_learning import DQNController


def load_test_problems():
    dataset_path = '/hada/domains/bbob_unconstrained/dataset.csv'
    test_problems = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'test':
                test_problems.append({
                    'problem_id': row['problem_id'],
                    'function': int(row['function']),
                    'instance': int(row['instance']),
                    'dimension': int(row['dimension']),
                    'opt_val': float(row['opt_val'])
                })
    return test_problems


def compute_ri_score(initial_gbest_f, gbest_f, opt_val):
    if abs(initial_gbest_f - opt_val) < 1e-12:
        return 1.0 if gbest_f <= opt_val + 1e-12 else 0.0
    score = (initial_gbest_f - gbest_f) / (initial_gbest_f - opt_val)
    return max(0.0, min(1.0, score))


def create_problem(problem_info):
    import cocoex
    suite = cocoex.Suite('bbob', '', 
                        f'dimensions: {problem_info["dimension"]} '
                        f'instance_indices:{problem_info["instance"]} '
                        f'function_indices: {problem_info["function"]}')
    problem = next(iter(suite))
    return problem


def evaluate_algorithm(problem_info, seed, max_evals=50000):
    """Evaluate the best gen algorithm on a single problem."""
    problem = create_problem(problem_info)
    opt_val = problem_info['opt_val']
    dim = problem_info['dimension']
    
    # Find DQN model for this seed
    model_path = None
    for lr in ['1e-02', '1e-03', '1e-04']:
        path = os.path.join(BEST_GEN_PATH, f'dqn_model_seed42_lr{lr}.pt')
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        model_path = os.path.join(BEST_GEN_PATH, 'dqn_model_seed42_lr1e-02.pt')
    
    # Create controller
    try:
        controller = DQNController(model_path=model_path, obs_dim=3, device='cpu')
    except:
        controller = None
    
    # Create optimizer
    optimizer = DEOptimizer(
        dim=dim,
        lower_bounds=problem.lower_bounds,
        upper_bounds=problem.upper_bounds,
        max_evals=max_evals,
        controller=controller
    )
    
    # Run optimization
    result, stats = optimizer.optimize(problem, seed=seed)
    
    gbest_f = stats.get('gbest_f', float('inf'))
    initial_gbest_f = stats.get('initial_gbest_f', gbest_f)
    
    # Compute RI score
    ri_score = compute_ri_score(initial_gbest_f, gbest_f, opt_val)
    
    return {
        'problem_id': problem_info['problem_id'],
        'function': problem_info['function'],
        'instance': problem_info['instance'],
        'dimension': problem_info['dimension'],
        'opt_val': opt_val,
        'seed': seed,
        'initial_gbest_f': initial_gbest_f,
        'final_gbest_f': gbest_f,
        'ri_score': ri_score,
        'evals': stats.get('evals', 0)
    }


def main():
    print("=" * 80)
    print("Best Generation Algorithm Evaluation | BBOB Unconstrained")
    print("=" * 80)
    
    # Load test problems
    print("\nLoading test problems...")
    test_problems = load_test_problems()
    print(f"  Test problems: {len(test_problems)}")
    
    # Seeds
    seeds = [42, 43, 44, 45, 46]
    print(f"\nSeeds: {seeds}")
    print(f"Max evals: 50000")
    
    # Evaluate
    all_results = []
    problem_scores = {}  # problem_id -> list of scores
    
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)
    
    start_time = time.time()
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        seed_scores = []
        
        for prob_info in test_problems:
            prob_id = prob_info['problem_id']
            
            try:
                result = evaluate_algorithm(prob_info, seed)
                all_results.append(result)
                seed_scores.append(result['ri_score'])
                
                if prob_id not in problem_scores:
                    problem_scores[prob_id] = []
                problem_scores[prob_id].append(result['ri_score'])
                
                print(f"  [{prob_id}] RI={result['ri_score']:.4f}, "
                      f"init={result['initial_gbest_f']:.4f}, "
                      f"final={result['final_gbest_f']:.6f}")
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                tb = traceback.format_exc()
                print(f"  [{prob_id}] ERROR: {error_msg}")
                print(f"  [{prob_id}] TRACEBACK:\n{tb}")
                if prob_id not in problem_scores:
                    problem_scores[prob_id] = []
                problem_scores[prob_id].append(0.0)
                seed_scores.append(0.0)
        
        seed_mean = np.mean(seed_scores) if seed_scores else 0.0
        print(f"  Seed {seed} Mean RI: {seed_mean:.4f}")
    
    elapsed = time.time() - start_time
    
    # Compute per-problem statistics
    print("\n" + "=" * 80)
    print("Per-Problem Results (mean+-std across 5 seeds):")
    print("=" * 80)
    
    problem_means = []
    for prob_info in test_problems:
        prob_id = prob_info['problem_id']
        scores = problem_scores.get(prob_id, [])
        
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            problem_means.append(mean_score)
            print(f"  {prob_id}: {mean_score:.4f}+-{std_score:.4f}")
        else:
            problem_means.append(0.0)
            print(f"  {prob_id}: N/A")
    
    # Overall statistics
    overall_mean = np.mean(problem_means) if problem_means else 0.0
    overall_std = np.std(problem_means) if problem_means else 0.0
    
    print("\n" + "=" * 80)
    print("Overall Results:")
    print("=" * 80)
    print(f"  Mean RI Score (across 16 problems): {overall_mean:.4f}+-{overall_std:.4f}")
    print(f"  Total time: {elapsed:.1f}s")
    
    # Save results
    output_dir = '/hada/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'best_gen_outcome.csv')
    with open(output_path, 'w') as f:
        f.write(f"{overall_mean:.4f}+-{overall_std:.4f}\n")
    
    print(f"\nResults saved to {output_path}")
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, 'best_gen_results.json')
    with open(detailed_path, 'w') as f:
        json.dump({
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'problem_scores': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                             for k, v in problem_scores.items()},
            'all_results': all_results
        }, f, indent=2)
    
    print(f"Detailed results saved to {detailed_path}")


if __name__ == '__main__':
    main()