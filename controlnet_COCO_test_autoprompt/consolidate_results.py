import os
import glob
import json
import shutil
from collections import defaultdict

# ====================== 配置与策略 ======================
TARGET_MODES = ["original_image", "pure_background"]
IMAGE_TYPES = ["baseline1", "baseline2", "baseline3", "method"]

# 汇聚后的统一输出文件夹名称
CONSOLIDATED_DIR_TEMPLATE = "results_all_{mode}"
# 汇总后的最终指标清单文件名
FINAL_MANIFEST_NAME = "eval_manifest_all.json"

def main():
    print("=" * 60)
    print("  开始启动实验结果自动化分拣、校验与全局汇总程序...")
    print("=" * 60)

    # ---------------------------------------------------------
    # Step 1: 自动扫描发现所有的批次输出文件夹
    # ---------------------------------------------------------
    all_items = os.listdir('.')
    chunk_dirs = [
        d for d in all_items 
        if os.path.isdir(d) and d.startswith("results_") and not d.startswith("results_all_")
    ]
    
    if not chunk_dirs:
        print("[-] 未在当前目录下检测到任何符合 `results_` 前缀的分批结果文件夹。")
        return

    print(f"[+] 成功识别到 {len(chunk_dirs)} 个分批结果目录。")

    # ---------------------------------------------------------
    # Step 2: 构建样本数据库，统计每张图的完整性
    # ---------------------------------------------------------
    # 结构: sample_registry[sample_name][mode][image_type] = src_file_path
    sample_registry = defaultdict(lambda: defaultdict(dict))

    for d in chunk_dirs:
        # 判断当前文件夹属于哪个 mode
        current_mode = None
        for m in TARGET_MODES:
            if d.endswith(m):
                current_mode = m
                break
        if not current_mode:
            continue  # 忽略命名不规范的文件夹
            
        for f in os.listdir(d):
            if f.endswith(".png"):
                base_part, _ = os.path.splitext(f)
                # 按照最右侧的下划线切分出样本名和消融类型 
                if "_" in base_part:
                    parts = base_part.rsplit("_", 1)
                    if len(parts) == 2:
                        sample_name, img_type = parts
                        if img_type in IMAGE_TYPES:
                            sample_registry[sample_name][current_mode][img_type] = os.path.join(d, f)

    total_discovered = len(sample_registry)
    print(f"[+] 全局数据集初步共发现 {total_discovered} 个独立样本。")

    # ---------------------------------------------------------
    # Step 3: 执行严格的完整性交叉校验 
    # ---------------------------------------------------------
    valid_samples = set()
    invalid_samples_count = 0

    for sample_name, modes_dict in sample_registry.items():
        is_complete = True
        # 一个完美的样本必须在两个 mode 下都具备完整的 4 张图
        for m in TARGET_MODES:
            if m not in modes_dict:
                is_complete = False
                break
            for t in IMAGE_TYPES:
                if t not in modes_dict[m]:
                    is_complete = False
                    break
        
        if is_complete:
            valid_samples.add(sample_name)
        else:
            invalid_samples_count += 1

    print(f"[+] 校验完毕: 完美通过成对测试的样本有 {len(valid_samples)} 个；因结果残缺被过滤的样本有 {invalid_samples_count} 个。")

    if not valid_samples:
        print("[-] 警告：没有找到任何一个具备完整 4 张结果图的样本，程序终止。")
        return

    # ---------------------------------------------------------
    # Step 4: 创统一总目录，拷贝有效结果图
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("[*] 正在创建全局统一汇总文件夹并迁移实体图片...")
    for m in TARGET_MODES:
        out_dir = CONSOLIDATED_DIR_TEMPLATE.format(mode=m)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)  # 清理旧的历史数据
        os.makedirs(out_dir, exist_ok=True)

    copy_count = 0
    for sample_name in sorted(valid_samples):
        for m in TARGET_MODES:
            out_dir = CONSOLIDATED_DIR_TEMPLATE.format(mode=m)
            for t in IMAGE_TYPES:
                src_path = sample_registry[sample_name][m][t]
                dst_name = f"{sample_name}_{t}.png"
                dst_path = os.path.join(out_dir, dst_name)
                shutil.copy2(src_path, dst_path)
                copy_count += 1
                
    print(f"[+] 图像数据汇聚完成：共计向新目录搬运图片 {copy_count} 张。")

    # ---------------------------------------------------------
    # Step 5: 清单文件汇总 (Merge & Filter eval_manifest)
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("[*] 正在扫描并汇总结算各区间的 `eval_manifest_*.json` 文件...")
    manifest_files = glob.glob("eval_manifest_*.json")
    # 排除掉汇总目标自身，防止自循环
    manifest_files = [f for f in manifest_files if f != FINAL_MANIFEST_NAME]

    all_manifest_entries = []
    for mf in manifest_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_manifest_entries.extend(data)
        except Exception as e:
            print(f" [-] 读取清单 {mf} 时发生错误，已跳过: {e}")

    # 过滤并修正清单条目中的路径信息
    final_manifest = []
    seen_keys = set() # 用于去除批次重叠或重复运行造成的冗余行
    
    for entry in all_manifest_entries:
        sample_name = entry.get("sample_name")
        mode = entry.get("mode")
        
        # 准入机制：只有被验证图像完整的样本才能入选最终清单
        if sample_name in valid_samples:
            unique_key = (sample_name, mode)
            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                
                # 将原本指向分批路径 (e.g. results_1to100_...) 的字段
                # 重新修正为指向汇总大目录的规范化路径，确保后续测评脚本可以一步到位访问
                target_mode_dir = CONSOLIDATED_DIR_TEMPLATE.format(mode=mode)
                entry["method_img_path"] = f"{target_mode_dir}/{sample_name}_method.png"
                final_manifest.append(entry)

    # 写入全局统一清单
    with open(FINAL_MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=4, ensure_ascii=False)

    print(f"[+] 清单汇总完成！已生成全量完整清单: `{FINAL_MANIFEST_NAME}` (共计条目: {len(final_manifest)} 行)")
    print("=" * 60)
    print("  [SUCCESS] 恭喜，所有分批实验结果已完美合并、清洗并归档完毕！")
    print("=" * 60)

if __name__ == "__main__":
    main()