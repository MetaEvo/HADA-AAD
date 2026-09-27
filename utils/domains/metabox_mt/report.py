"""
MetaBox MT Domain - Report Generation
多任务黑箱优化的报告生成
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt


def generate_report(output_dir: str, predictions: pd.DataFrame, mode: str = 'train') -> str:
    """
    生成多任务优化评估报告
    
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
    print(f"MetaBox MT Optimization Report ({mode.upper()})")
    print(f"{'='*80}\n")
    
    with open(report_path, 'w') as f:
        f.write(f"# MetaBox MT Optimization Report ({mode.upper()})\n\n")
        
        # 总体统计
        f.write("## Overall Statistics\n\n")
        print("## Overall Statistics\n")
        
        total_problems = len(predictions)
        f.write(f"- **Total Problems**: {total_problems}\n")
        print(f"- Total Problems: {total_problems}")
        
        # 解析预测结果
        scores = []
        task_scores = []
        ktes = []
        n_tasks_list = []
        task_gbest_f_list = []
        similarities = []
        
        for _, row in predictions.iterrows():
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 7:
                    scores.append(float(parts[0]))
                    task_scores.append(float(parts[1]))
                    ktes.append(float(parts[2]))
                    n_tasks_list.append(int(float(parts[3])))
                    # 解析 JSON 格式的 task_gbest_f
                    try:
                        task_gbest_f = json.loads(parts[5])
                        task_gbest_f_list.append(task_gbest_f)
                    except:
                        task_gbest_f_list.append([])
                    similarities.append(float(parts[6]))
                elif len(parts) >= 4:
                    # 兼容旧格式
                    scores.append(float(parts[0]))
                    task_scores.append(float(parts[1]))
                    ktes.append(float(parts[2]))
                    n_tasks_list.append(int(float(parts[3])))
                    task_gbest_f_list.append([])
                    similarities.append(0.5)
            except Exception as e:
                print(f"  Warning: Failed to parse prediction for {row.get('problem_id', 'unknown')}: {e}")
                continue
        
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            avg_task = np.mean(task_scores)
            avg_kte = np.mean(ktes)
            avg_n_tasks = np.mean(n_tasks_list)
            
            f.write(f"- **Average Score**: {avg_score:.4f} (±{std_score:.4f})\n")
            f.write(f"- **Score Range**: [{min_score:.4f}, {max_score:.4f}]\n")
            f.write(f"- **Average Task Score**: {avg_task:.4f}\n")
            f.write(f"- **Average Knowledge Transfer**: {avg_kte:.4f}\n")
            f.write(f"- **Average Tasks per Problem**: {avg_n_tasks:.1f}\n\n")
            
            print(f"- Average Score: {avg_score:.4f} (±{std_score:.4f})")
            print(f"- Score Range: [{min_score:.4f}, {max_score:.4f}]")
            print(f"- Average Task Score: {avg_task:.4f}")
            print(f"- Average Knowledge Transfer: {avg_kte:.4f}")
            print(f"- Average Tasks per Problem: {avg_n_tasks:.1f}\n")
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
                suites[suite] = {'scores': [], 'task_scores': [], 'ktes': [], 'n_tasks': []}
            
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 4:
                    suites[suite]['scores'].append(float(parts[0]))
                    suites[suite]['task_scores'].append(float(parts[1]))
                    suites[suite]['ktes'].append(float(parts[2]))
                    suites[suite]['n_tasks'].append(int(float(parts[3])))
            except:
                continue
        
        for suite, data in suites.items():
            if data['scores']:
                f.write(f"### {suite.upper()}\n")
                f.write(f"- Problems: {len(data['scores'])}\n")
                f.write(f"- Avg Score: {np.mean(data['scores']):.4f} (±{np.std(data['scores']):.4f})\n")
                f.write(f"- Avg Task Score: {np.mean(data['task_scores']):.4f}\n")
                f.write(f"- Avg Knowledge Transfer: {np.mean(data['ktes']):.4f}\n")
                f.write(f"- Avg Tasks: {np.mean(data['n_tasks']):.1f}\n\n")
                
                print(f"### {suite.upper()}")
                print(f"  - Problems: {len(data['scores'])}")
                print(f"  - Avg Score: {np.mean(data['scores']):.4f} (±{np.std(data['scores']):.4f})")
                print(f"  - Avg Task Score: {np.mean(data['task_scores']):.4f}")
                print(f"  - Avg Knowledge Transfer: {np.mean(data['ktes']):.4f}")
                print(f"  - Avg Tasks: {np.mean(data['n_tasks']):.1f}\n")
        
        # 按任务数量分组
        f.write("## Statistics by Number of Tasks\n\n")
        print("## Statistics by Number of Tasks\n")
        
        task_groups = {}
        for _, row in predictions.iterrows():
            n_tasks = row.get('n_tasks', 50)
            if n_tasks not in task_groups:
                task_groups[n_tasks] = {'scores': [], 'task_scores': [], 'ktes': []}
            
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 4:
                    task_groups[n_tasks]['scores'].append(float(parts[0]))
                    task_groups[n_tasks]['task_scores'].append(float(parts[1]))
                    task_groups[n_tasks]['ktes'].append(float(parts[2]))
            except:
                continue
        
        f.write("| Tasks | Problems | Avg Score | Avg Task Score | Avg KTE |\n")
        f.write("|-------|----------|-----------|----------------|---------|\n")
        print("| Tasks | Problems | Avg Score | Avg Task Score | Avg KTE |")
        print("|-------|----------|-----------|----------------|---------|")
        
        for n_tasks in sorted(task_groups.keys()):
            data = task_groups[n_tasks]
            if data['scores']:
                f.write(f"| {n_tasks} | {len(data['scores'])} | {np.mean(data['scores']):.4f} | {np.mean(data['task_scores']):.4f} | {np.mean(data['ktes']):.4f} |\n")
                print(f"| {n_tasks} | {len(data['scores'])} | {np.mean(data['scores']):.4f} | {np.mean(data['task_scores']):.4f} | {np.mean(data['ktes']):.4f} |")
        
        f.write("\n")
        print()
        
        # 详细结果
        f.write("## Detailed Results\n\n")
        f.write("| Problem | Suite | Instance | Dimension | Tasks | Score | Task Score | KTE | Task gbest_f | Similarity |\n")
        f.write("|---------|-------|----------|-----------|-------|-------|------------|-----|--------------|------------|\n")
        
        print("## Detailed Results\n")
        print("| Problem | Suite | Instance | Dimension | Tasks | Score | Task Score | KTE | Task gbest_f | Similarity |")
        print("|---------|-------|----------|-----------|-------|-------|------------|-----|--------------|------------|")
        
        for idx, (_, row) in enumerate(predictions.iterrows()):
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 7:
                    score = float(parts[0])
                    task_score = float(parts[1])
                    kte = float(parts[2])
                    n_tasks = int(float(parts[3]))
                    # 解析 task_gbest_f
                    try:
                        task_gbest_f = json.loads(parts[5])
                        gbest_f_str = ", ".join([f"{f:.2f}" for f in task_gbest_f])
                    except:
                        gbest_f_str = "-"
                    similarity = float(parts[6])
                    
                    line = f"| {row.get('problem_id', 'N/A')} | {row.get('suite', 'N/A')} | {row.get('instance', 'N/A')} | {row.get('dimension', 'N/A')} | {n_tasks} | {score:.4f} | {task_score:.4f} | {kte:.4f} | {gbest_f_str} | {similarity:.4f} |"
                    f.write(line + "\n")
                    print(line)
                elif len(parts) >= 4:
                    score = float(parts[0])
                    task_score = float(parts[1])
                    kte = float(parts[2])
                    n_tasks = int(float(parts[3]))
                    
                    line = f"| {row.get('problem_id', 'N/A')} | {row.get('suite', 'N/A')} | {row.get('instance', 'N/A')} | {row.get('dimension', 'N/A')} | {n_tasks} | {score:.4f} | {task_score:.4f} | {kte:.4f} | - | - |"
                    f.write(line + "\n")
                    print(line)
            except Exception as e:
                error_line = f"| {row.get('problem_id', 'N/A')} | ERROR | - | - | - | - | - | - | - | - |"
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
            avg_kte = np.mean(ktes)
            
            if avg_score > 0.8:
                conclusion = "✅ **Excellent performance**: The PSO algorithm shows strong multi-task optimization capability."
            elif avg_score > 0.6:
                conclusion = "✓ **Good performance**: The algorithm performs well but has room for improvement."
            elif avg_score > 0.4:
                conclusion = "⚠ **Moderate performance**: The algorithm needs significant improvements."
            else:
                conclusion = "❌ **Poor performance**: The algorithm requires major modifications."
            
            f.write(conclusion + "\n")
            print(conclusion)
            
            f.write("\n### Knowledge Transfer Analysis\n\n")
            print("\n### Knowledge Transfer Analysis\n")
            
            if avg_kte > 1.1:
                kte_conclusion = "✅ **Positive knowledge transfer**: Multi-task learning is beneficial."
            elif avg_kte > 0.9:
                kte_conclusion = "✓ **Neutral knowledge transfer**: No significant benefit or harm."
            else:
                kte_conclusion = "⚠ **Negative knowledge transfer**: Tasks may be interfering with each other."
            
            f.write(kte_conclusion + "\n")
            print(kte_conclusion)
            
            f.write("\n### Recommendations\n\n")
            print("\n### Recommendations\n")
            
            if avg_kte < 0.9:
                rec = "- Implement task similarity measurement"
                f.write(rec + "\n")
                print(rec)
                rec = "- Use adaptive parameter sharing"
                f.write(rec + "\n")
                print(rec)
                rec = "- Consider task clustering or grouping"
                f.write(rec + "\n")
                print(rec)
            
            rec = "- Implement specialized multi-task PSO variants"
            f.write(rec + "\n")
            print(rec)
            rec = "- Add task-specific parameter adaptation"
            f.write(rec + "\n")
            print(rec)
            rec = "- Optimize hyperparameters for different task numbers"
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
            'mean_task_score': float(avg_task),
            'mean_kte': float(avg_kte),
            'mean_n_tasks': float(avg_n_tasks),
            'total_problems': total_problems,
            'valid_predictions': len(scores)
        }
    else:
        report_json = {
            'score': 0.0,
            'std_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'mean_task_score': 0.0,
            'mean_kte': 0.0,
            'mean_n_tasks': 0.0,
            'total_problems': total_problems,
            'valid_predictions': 0
        }
    
    json_path = os.path.join(output_dir, 'report.json')
    with open(json_path, 'w') as f:
        json.dump(report_json, f, indent=2)
    print(f"JSON report saved to: {json_path}")
    
    return report_path


def plot_task_performance(output_dir: str, predictions: pd.DataFrame, mode: str = 'train'):
    """
    绘制任务性能图
    
    Args:
        output_dir: 输出目录
        predictions: 预测结果 DataFrame
        mode: 'train' 或 'test'
    """
    try:
        # 按任务数量分组绘制
        task_groups = {}
        for _, row in predictions.iterrows():
            n_tasks = row.get('n_tasks', 50)
            if n_tasks not in task_groups:
                task_groups[n_tasks] = {'scores': [], 'ktes': []}
            
            try:
                parts = str(row['prediction']).split('|')
                if len(parts) >= 4:
                    task_groups[n_tasks]['scores'].append(float(parts[0]))
                    task_groups[n_tasks]['ktes'].append(float(parts[2]))
            except:
                continue
        
        if not task_groups:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 分数 vs 任务数
        n_tasks_list = sorted(task_groups.keys())
        avg_scores = [np.mean(task_groups[n]['scores']) for n in n_tasks_list]
        
        ax1.plot(n_tasks_list, avg_scores, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Tasks')
        ax1.set_ylabel('Average Score')
        ax1.set_title('Performance vs Number of Tasks')
        ax1.grid(True, alpha=0.3)
        
        # 知识迁移效率 vs 任务数
        avg_ktes = [np.mean(task_groups[n]['ktes']) for n in n_tasks_list]
        
        ax2.plot(n_tasks_list, avg_ktes, 's-', linewidth=2, markersize=8, color='orange')
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Number of Tasks')
        ax2.set_ylabel('Knowledge Transfer Efficiency')
        ax2.set_title('KTE vs Number of Tasks')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'task_performance_{mode}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error plotting task performance: {e}")


def generate_summary(output_dir: str, all_results: Dict[str, pd.DataFrame]):
    """
    生成汇总报告
    
    Args:
        output_dir: 输出目录
        all_results: 各 domain 的结果字典
    """
    summary_path = os.path.join(output_dir, 'summary.md')
    
    with open(summary_path, 'w') as f:
        f.write("# MetaBox MT Optimization Summary\n\n")
        
        for domain, predictions in all_results.items():
            f.write(f"## {domain}\n\n")
            
            if predictions.empty:
                f.write("No results available.\n\n")
                continue
            
            # 计算统计信息
            scores = []
            ktes = []
            for _, row in predictions.iterrows():
                try:
                    parts = str(row['prediction']).split('|')
                    if len(parts) >= 4:
                        scores.append(float(parts[0]))
                        ktes.append(float(parts[2]))
                except:
                    continue
            
            if scores:
                f.write(f"- **Problems**: {len(predictions)}\n")
                f.write(f"- **Average Score**: {np.mean(scores):.4f}\n")
                f.write(f"- **Average KTE**: {np.mean(ktes):.4f}\n")
                f.write(f"- **Std Score**: {np.std(scores):.4f}\n\n")
    
    return summary_path