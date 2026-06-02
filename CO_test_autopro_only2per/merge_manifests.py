import json
import glob
import os
import re

def extract_start_idx(filename):
    """从文件名中提取起始索引用于排序"""
    match = re.search(r'eval_manifest_(\d+)_', filename)
    return int(match.group(1)) if match else 0

# 获取所有 manifest 文件（排除合并后的输出文件）
manifest_files = glob.glob("eval_manifest_*_*.json")
manifest_files = [f for f in manifest_files if f != "eval_manifest_all.json"]
manifest_files.sort(key=extract_start_idx)

all_records = []
for filepath in manifest_files:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            all_records.extend(data)
        else:
            print(f"警告：{filepath} 不是列表格式，已跳过")
    print(f"已加载 {filepath}，包含 {len(data) if isinstance(data, list) else 0} 条记录")

# 写入合并后的文件
output_path = "eval_manifest_all.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_records, f, indent=4, ensure_ascii=False)

print(f"合并完成！共 {len(all_records)} 条记录，保存至 {output_path}")