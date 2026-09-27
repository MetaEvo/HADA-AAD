#!/usr/bin/env python3
"""
提取最佳代际的所有 all_patch.diff 并在对应 generate 文件夹下创建新的 metabbo。
"""

import os
import sys
import json
import shutil
import subprocess


def get_score_from_metadata(gen_dir, domain="metabox_mo"):
    metadata_file = os.path.join(gen_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    report_path = os.path.join(gen_dir, f"{domain}_eval", "report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)
            if "score" in report:
                return report["score"]
            if "mean_score" in report:
                return report["mean_score"]
    
    return None


def find_best_generation(output_dir, domain="metabox_mo"):
    best_genid = None
    best_score = -float('inf')
    
    for item in sorted(os.listdir(output_dir)):
        if not item.startswith("gen_"):
            continue
        
        genid = item.replace("gen_", "")
        if genid == "initial":
            continue
        
        gen_dir = os.path.join(output_dir, item)
        score = get_score_from_metadata(gen_dir, domain)
        
        if score is not None and score > best_score:
            best_score = score
            best_genid = genid
    
    return best_genid, best_score


def get_all_ancestor_patches(output_dir, genid, domain="metabox_mo"):
    patches = []
    visited = set()
    
    current_genid = genid
    while current_genid and current_genid != "initial" and current_genid not in visited:
        visited.add(current_genid)
        
        gen_dir = os.path.join(output_dir, f"gen_{current_genid}")
        metadata_file = os.path.join(gen_dir, "metadata.json")
        
        if not os.path.exists(metadata_file):
            break
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        patch_path = None
        eval_folder = os.path.join(gen_dir, f"{domain}_eval", "agent_evals")
        candidate = os.path.join(eval_folder, "all_patch.diff")
        if os.path.exists(candidate):
            patch_path = candidate
        
        if patch_path:
            patches.append(patch_path)
            print(f"  Found patch: {patch_path}")
        
        current_genid = metadata.get("parent_genid")
    
    patches.reverse()
    return patches


def apply_patches_to_new_cocoex(patches, src_cocoex_dir, dst_cocoex_dir):
    """复制原始 metabbo 到新位置并应用 patches"""
    if not patches:
        print("No patches to apply.")
        return False
    
    # 复制原始目录到新位置
    if os.path.exists(dst_cocoex_dir):
        shutil.rmtree(dst_cocoex_dir)
    shutil.copytree(src_cocoex_dir, dst_cocoex_dir)
    print(f"Copied {src_cocoex_dir} to {dst_cocoex_dir}")
    
    # 应用每个 patch（只提取 metabbo 部分）
    for patch_file in patches:
        print(f"Applying patch: {os.path.basename(patch_file)}")
        
        with open(patch_file, 'r') as f:
            patch_content = f.read()
        
        # 只保留 metabbo 相关的 diff 块
        lines = patch_content.split('\n')
        filtered_lines = []
        in_cocoex = False
        
        for line in lines:
            if line.startswith('diff --git'):
                in_cocoex = 'metabbo/' in line
            if in_cocoex:
                filtered_lines.append(line)
        
        if not filtered_lines:
            print("  No metabbo changes, skipping")
            continue
        
        filtered_patch = '\n'.join(filtered_lines)
        
        # 使用 -p2 去掉 a/metabbo/ 前缀
        result = subprocess.run(
            ['patch', '-p2', '-f'],
            cwd=dst_cocoex_dir,
            input=filtered_patch,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  Successfully applied")
        else:
            print(f"  Warning: patch returned {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_best_gen.py <output_dir> [domain]")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) > 2 else "metabox_mo"
    
    print(f"Searching for best generation in {output_dir}...")
    best_genid, best_score = find_best_generation(output_dir, domain)
    
    if best_genid is None:
        print("No valid generation found with scores.")
        sys.exit(1)
    
    print(f"Best generation: gen_{best_genid} (score: {best_score})")
    
    print(f"\nCollecting patches from gen_{best_genid} and ancestors...")
    patches = get_all_ancestor_patches(output_dir, best_genid, domain)
    
    if not patches:
        print("No all_patch.diff files found.")
        sys.exit(1)
    
    print(f"\nFound {len(patches)} patches total.")
    
    # 源目录：项目根目录下的原始 metabbo
    src_cocoex_dir = os.path.join(os.getcwd(), "metabbo")
    if not os.path.exists(src_cocoex_dir):
        print(f"\nError: Source metabbo not found at {src_cocoex_dir}")
        sys.exit(1)
    
    # 目标目录：在 generate 文件夹下创建新的 metabbo
    dst_cocoex_dir = os.path.join(output_dir, "metabbo_best_gen")
    
    print(f"\nCreating new metabbo at {dst_cocoex_dir}...")
    apply_patches_to_new_cocoex(patches, src_cocoex_dir, dst_cocoex_dir)
    
    print(f"\nDone! Final code is in {dst_cocoex_dir}")


if __name__ == "__main__":
    main()