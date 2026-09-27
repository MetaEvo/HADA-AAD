"""
MetaBox MO Domain - Report Generation
多目标黑箱优化的报告生成
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt


def generate_report(output_dir: str, predictions: pd.DataFrame, mode: str = 'train') -> str:
    """
    生成多目标优化评估报告
    
    Args:
        output_dir: 输出目录
        predictions: 预测结果 DataFrame
        mode: 'train' 或 'test'
    
    Returns:
        报告文件路径
    """
    report_path = os.path.join(output_dir, f'report_{mode}.md')
    
    # 同时输出到控制台
    print(f"\n{'='*80}")
    print(f"MetaBox MO Optimization Report ({mode.upper()})")
    print(f"{'='*80}\n")
    
    with open(report_path, 'w') as f:
        f.write(f"# MetaBox MO Optimization Report ({mode.upper()})\n\n")
        
        # 总体统计
        f.write("## Overall Statistics\n\n")
        print("## Overall Statistics\n")
        
        total_problems = len(predictions)
        f.write(f"- **Total Problems**: {total_problems}\n")
        print(f"- Total Problems: {total_problems}")
        
        # 解析预测结果
        scores = []
        hypervolumes = []
        igds = []
        gds = []
        spacings = []
        n_solutions_list = []
        
        for _, row in predictions.iterrows():
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 6:
                    scores.append(float(parts[0]))
                    hypervolumes.append(float(parts[1]))
                    igds.append(float(parts[2]))
                    gds.append(float(parts[3]))
                    spacings.append(float(parts[4]))
                    n_solutions_list.append(int(float(parts[5])))
                elif len(parts) >= 4:
                    # 兼容旧格式
                    scores.append(float(parts[0]))
                    hypervolumes.append(float(parts[1]))
                    igds.append(float(parts[2]))
                    gds.append(float('inf'))
                    spacings.append(0.0)
                    n_solutions_list.append(int(float(parts[3])))
            except Exception as e:
                print(f"  Warning: Failed to parse prediction for {row.get('problem_id', 'unknown')}: {e}")
                continue
        
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            avg_hv = np.mean(hypervolumes)
            avg_igd = np.mean(igds)
            avg_gd = np.mean(gds) if gds else float('inf')
            avg_spacing = np.mean(spacings) if spacings else 0.0
            avg_n_sol = np.mean(n_solutions_list)
            
            f.write(f"- **Average Score**: {avg_score:.4f} (±{std_score:.4f})\n")
            f.write(f"- **Score Range**: [{min_score:.4f}, {max_score:.4f}]\n")
            f.write(f"- **Average Hypervolume**: {avg_hv:.4f}\n")
            f.write(f"- **Average IGD**: {avg_igd:.4f}\n")
            f.write(f"- **Average GD**: {avg_gd:.4f}\n")
            f.write(f"- **Average Spacing**: {avg_spacing:.4f}\n")
            f.write(f"- **Average Pareto Solutions**: {avg_n_sol:.1f}\n\n")
            
            print(f"- Average Score: {avg_score:.4f} (±{std_score:.4f})")
            print(f"- Score Range: [{min_score:.4f}, {max_score:.4f}]")
            print(f"- Average Hypervolume: {avg_hv:.4f}")
            print(f"- Average IGD: {avg_igd:.4f}")
            print(f"- Average GD: {avg_gd:.4f}")
            print(f"- Average Spacing: {avg_spacing:.4f}")
            print(f"- Average Pareto Solutions: {avg_n_sol:.1f}\n")
        else:
            f.write("- **No valid predictions found**\n\n")
            print("- No valid predictions found!\n")
        
        # 按问题套件分组统计
        f.write("## Statistics by Suite\n\n")
        print("## Statistics by Suite\n")
        
        suites = {}
        for _, row in predictions.iterrows():
            suite = row.get('suite', 'unknown')
            if suite not in suites:
                suites[suite] = {'scores': [], 'hvs': [], 'igds': [], 'gds': [], 'spacings': [], 'n_sols': []}
            
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 6:
                    suites[suite]['scores'].append(float(parts[0]))
                    suites[suite]['hvs'].append(float(parts[1]))
                    suites[suite]['igds'].append(float(parts[2]))
                    suites[suite]['gds'].append(float(parts[3]))
                    suites[suite]['spacings'].append(float(parts[4]))
                    suites[suite]['n_sols'].append(int(float(parts[5])))
                elif len(parts) >= 4:
                    suites[suite]['scores'].append(float(parts[0]))
                    suites[suite]['hvs'].append(float(parts[1]))
                    suites[suite]['igds'].append(float(parts[2]))
                    suites[suite]['gds'].append(float('inf'))
                    suites[suite]['spacings'].append(0.0)
                    suites[suite]['n_sols'].append(int(float(parts[3])))
            except:
                continue
        
        for suite, data in suites.items():
            if data['scores']:
                f.write(f"### {suite.upper()}\n")
                f.write(f"- Problems: {len(data['scores'])}\n")
                f.write(f"- Avg Score: {np.mean(data['scores']):.4f} (±{np.std(data['scores']):.4f})\n")
                f.write(f"- Avg Hypervolume: {np.mean(data['hvs']):.4f}\n")
                f.write(f"- Avg IGD: {np.mean(data['igds']):.4f}\n")
                f.write(f"- Avg GD: {np.mean(data['gds']):.4f}\n")
                f.write(f"- Avg Spacing: {np.mean(data['spacings']):.4f}\n")
                f.write(f"- Avg Pareto Solutions: {np.mean(data['n_sols']):.1f}\n\n")
                
                print(f"### {suite.upper()}")
                print(f"  - Problems: {len(data['scores'])}")
                print(f"  - Avg Score: {np.mean(data['scores']):.4f} (±{np.std(data['scores']):.4f})")
                print(f"  - Avg Hypervolume: {np.mean(data['hvs']):.4f}")
                print(f"  - Avg IGD: {np.mean(data['igds']):.4f}")
                print(f"  - Avg GD: {np.mean(data['gds']):.4f}")
                print(f"  - Avg Spacing: {np.mean(data['spacings']):.4f}")
                print(f"  - Avg Pareto Solutions: {np.mean(data['n_sols']):.1f}\n")
        
        # 按维度分组统计
        f.write("## Statistics by Dimension\n\n")
        print("## Statistics by Dimension\n")
        
        dims = {}
        for _, row in predictions.iterrows():
            dim = row.get('dimension', 'unknown')
            if dim not in dims:
                dims[dim] = {'scores': [], 'hvs': [], 'igds': [], 'gds': [], 'spacings': []}
            
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 6:
                    dims[dim]['scores'].append(float(parts[0]))
                    dims[dim]['hvs'].append(float(parts[1]))
                    dims[dim]['igds'].append(float(parts[2]))
                    dims[dim]['gds'].append(float(parts[3]))
                    dims[dim]['spacings'].append(float(parts[4]))
                elif len(parts) >= 4:
                    dims[dim]['scores'].append(float(parts[0]))
                    dims[dim]['hvs'].append(float(parts[1]))
                    dims[dim]['igds'].append(float(parts[2]))
                    dims[dim]['gds'].append(float('inf'))
                    dims[dim]['spacings'].append(0.0)
            except:
                continue
        
        f.write("| Dimension | Problems | Avg Score | Avg HV | Avg IGD | Avg GD | Avg Spacing |\n")
        f.write("|-----------|----------|-----------|--------|---------|--------|-------------|\n")
        print("| Dimension | Problems | Avg Score | Avg HV | Avg IGD | Avg GD | Avg Spacing |")
        print("|-----------|----------|-----------|--------|---------|--------|-------------|")
        
        for dim in sorted(dims.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            data = dims[dim]
            if data['scores']:
                f.write(f"| {dim} | {len(data['scores'])} | {np.mean(data['scores']):.4f} | {np.mean(data['hvs']):.4f} | {np.mean(data['igds']):.4f} | {np.mean(data['gds']):.4f} | {np.mean(data['spacings']):.4f} |\n")
                print(f"| {dim} | {len(data['scores'])} | {np.mean(data['scores']):.4f} | {np.mean(data['hvs']):.4f} | {np.mean(data['igds']):.4f} | {np.mean(data['gds']):.4f} | {np.mean(data['spacings']):.4f} |")
        
        f.write("\n")
        print()
        
        # 详细结果
        f.write("## Detailed Results\n\n")
        f.write("| Problem | Suite | Function | Dimension | Objectives | Score | Hypervolume | IGD | GD | Spacing | N Solutions |\n")
        f.write("|---------|-------|----------|-----------|------------|-------|-------------|-----|-----|---------|-------------|\n")
        
        print("## Detailed Results\n")
        print("| Problem | Suite | Function | Dimension | Objectives | Score | Hypervolume | IGD | GD | Spacing | N Solutions |")
        print("|---------|-------|----------|-----------|------------|-------|-------------|-----|-----|---------|-------------|")
        
        for _, row in predictions.iterrows():
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 6:
                    score = float(parts[0])
                    hv = float(parts[1])
                    igd = float(parts[2])
                    gd = float(parts[3])
                    spacing = float(parts[4])
                    n_sol = int(float(parts[5]))
                    
                    line = f"| {row.get('problem_id', 'N/A')} | {row.get('suite', 'N/A')} | {row.get('function', 'N/A')} | {row.get('dimension', 'N/A')} | {row.get('n_objectives', 'N/A')} | {score:.4f} | {hv:.4f} | {igd:.4f} | {gd:.4f} | {spacing:.4f} | {n_sol} |"
                    f.write(line + "\n")
                    print(line)
                elif len(parts) >= 4:
                    score = float(parts[0])
                    hv = float(parts[1])
                    igd = float(parts[2])
                    n_sol = int(float(parts[3]))
                    
                    line = f"| {row.get('problem_id', 'N/A')} | {row.get('suite', 'N/A')} | {row.get('function', 'N/A')} | {row.get('dimension', 'N/A')} | {row.get('n_objectives', 'N/A')} | {score:.4f} | {hv:.4f} | {igd:.4f} | - | - | {n_sol} |"
                    f.write(line + "\n")
                    print(line)
            except Exception as e:
                error_line = f"| {row.get('problem_id', 'N/A')} | ERROR | - | - | - | - | - | - | - | - | - |"
                f.write(error_line + "\n")
                print(f"Error parsing row: {e}")
                continue
        
        f.write("\n")
        print()
        
        # 结论和建议
        f.write("## Conclusions\n\n")
        print("## Conclusions\n")
        
        if scores:
            avg_score = np.mean(scores)
            avg_igd = np.mean(igds)
            avg_hv = np.mean(hypervolumes)
            
            if avg_score > 0.8:
                conclusion = "✅ **Excellent performance**: The PSO algorithm shows strong multi-objective optimization capability."
            elif avg_score > 0.6:
                conclusion = "✓ **Good performance**: The algorithm performs well but has room for improvement."
            elif avg_score > 0.4:
                conclusion = "⚠ **Moderate performance**: The algorithm needs significant improvements."
            else:
                conclusion = "❌ **Poor performance**: The algorithm requires major modifications."
            
            f.write(conclusion + "\n")
            print(conclusion)
            
            f.write("\n### Recommendations\n\n")
            print("\n### Recommendations\n")
            
            if avg_igd > 5.0:
                rec = "- Improve convergence to the true Pareto front"
                f.write(rec + "\n")
                print(rec)
                rec = "- Consider better leader selection strategies"
                f.write(rec + "\n")
                print(rec)
            
            if avg_hv < 10.0:
                rec = "- Increase diversity of solutions"
                f.write(rec + "\n")
                print(rec)
                rec = "- Implement better archive mechanisms"
                f.write(rec + "\n")
                print(rec)
            
            rec = "- Consider using specialized multi-objective PSO variants"
            f.write(rec + "\n")
            print(rec)
            rec = "- Tune hyperparameters for different problem characteristics"
            f.write(rec + "\n")
            print(rec)
        else:
            f.write("No valid results to analyze.\n")
            print("No valid results to analyze.")
    
    print(f"\n{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*80}\n")
    
    # 同时生成 report.json 供 generate_loop 读取分数
    if scores:
        report_json = {
            'score': float(avg_score),
            'std_score': float(std_score),
            'min_score': float(min_score),
            'max_score': float(max_score),
            'mean_hypervolume': float(avg_hv),
            'mean_igd': float(avg_igd),
            'mean_n_solutions': float(avg_n_sol),
            'total_problems': total_problems,
            'valid_predictions': len(scores)
        }
    else:
        report_json = {
            'score': 0.0,
            'std_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'mean_hypervolume': 0.0,
            'mean_igd': float('inf'),
            'mean_n_solutions': 0.0,
            'total_problems': total_problems,
            'valid_predictions': 0
        }
    
    json_path = os.path.join(output_dir, 'report.json')
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    print(f"JSON report saved to: {json_path}")
    
    return report_path


def plot_pareto_fronts(output_dir: str, predictions: pd.DataFrame, mode: str = 'train'):
    """
    绘制 Pareto 前沿图
    
    Args:
        output_dir: 输出目录
        predictions: 预测结果 DataFrame
        mode: 'train' 或 'test'
    """
    try:
        # 收集2目标问题的Pareto前沿
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for _, row in predictions.iterrows():
            if plot_idx >= 6:
                break
            
            n_obj = row.get('n_objectives', 2)
            if n_obj != 2:
                continue
            
            try:
                # 解析 Pareto 前沿
                parts = str(row['prediction']).split('|')
                if len(parts) >= 5:
                    pareto_front = json.loads(parts[4])
                    pareto_front = np.array(pareto_front)
                    
                    if len(pareto_front) > 0:
                        ax = axes[plot_idx]
                        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                                  alpha=0.6, s=30)
                        ax.set_xlabel('f1')
                        ax.set_ylabel('f2')
                        ax.set_title(f"{row['problem_id']}")
                        ax.grid(True, alpha=0.3)
                        plot_idx += 1
            except:
                continue
        
        # 隐藏未使用的子图
        for i in range(plot_idx, 6):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pareto_fronts_{mode}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting Pareto fronts: {e}")


def generate_summary(output_dir: str, all_results: Dict[str, pd.DataFrame]):
    """
    生成汇总报告
    
    Args:
        output_dir: 输出目录
        all_results: 各 domain 的结果字典
    """
    summary_path = os.path.join(output_dir, 'summary.md')
    
    with open(summary_path, 'w') as f:
        f.write("# MetaBox MO Optimization Summary\n\n")
        
        for domain, predictions in all_results.items():
            f.write(f"## {domain}\n\n")
            
            if predictions.empty:
                f.write("No results available.\n\n")
                continue
            
            # 计算统计信息
            scores = []
            for _, row in predictions.iterrows():
                try:
                    parts = str(row['prediction']).split('|')
                    if len(parts) >= 1:
                        scores.append(float(parts[0]))
                except:
                    continue
            
            if scores:
                f.write(f"- **Problems**: {len(predictions)}\n")
                f.write(f"- **Average Score**: {np.mean(scores):.4f}\n")
                f.write(f"- **Std Score**: {np.std(scores):.4f}\n")
                f.write(f"- **Min Score**: {np.min(scores):.4f}\n")
                f.write(f"- **Max Score**: {np.max(scores):.4f}\n\n")
    
    return summary_path