# 检查所有JSON文件中的RGB值是否重复
# 输入目录，输出检查结果
import os
import json
import glob

def check_json_no_duplicate_rgb(input_dir):
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print("未找到 JSON 文件")
        return
        
    total_pass = 0
    total_fail = 0
    
    for json_file in json_files:
        basename = os.path.basename(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        rgb_list = []
        
        if isinstance(data, list):
            for item in data:
                if 'rgb' in item:
                    rgb_list.append(tuple(item['rgb']))
        elif isinstance(data, dict):
            for key, item in data.items():
                if 'rgb' in item:
                    rgb_list.append(tuple(item['rgb']))
                    
        seen = set()
        duplicates = set()
        
        for rgb in rgb_list:
            if rgb in seen:
                duplicates.add(rgb)
            seen.add(rgb)
            
        if duplicates:
            total_fail += 1
            print(f"[FAIL] {basename}: {len(rgb_list)} 条, 发现 {len(duplicates)} 个重复 RGB: {[list(d) for d in duplicates][:5]}")
        else:
            total_pass += 1
            print(f"[OK]   {basename}: {len(rgb_list)} 条, 无重复")
            
    print("-" * 50)
    print(f"通过: {total_pass}, 失败: {total_fail}")
    
    if total_fail == 0:
        print("所有 JSON 文件 RGB 均无重复！")


if __name__ == "__main__":
    # 检查 step_colored 目录下由 color_step_faces.py 生成的 _colors.json
    check_json_no_duplicate_rgb(r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\step_colored")
