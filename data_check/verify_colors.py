import os
import json
import glob

def verify_extracted_colors():
    step_colored_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\step_colored"
    extracted_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\extracted_colors"
    
    json_files = glob.glob(os.path.join(step_colored_dir, "*_colors.json"))
    
    total_extracted = 0
    total_matched = 0
    total_unmatched = 0
    
    for orig_json in json_files:
        basename = os.path.basename(orig_json).replace("_colors.json", "")
        # The extracted dir has a subfolder named like: 510_CHANGHANGJIETOU_3_stp_ortho_front
        extracted_folder = os.path.join(extracted_dir, basename + "_ortho_front")
        extracted_json = os.path.join(extracted_folder, "coordinates.json")
        
        if not os.path.exists(extracted_json):
            continue
            
        # Load original colors
        with open(orig_json, 'r', encoding='utf-8') as f:
            orig_data = json.load(f)
        orig_rgbs = set(tuple(item['rgb']) for item in orig_data)
        
        # Load extracted colors
        with open(extracted_json, 'r', encoding='utf-8') as f:
            ext_data = json.load(f)
        ext_rgbs = set(tuple(item['rgb']) for item in ext_data.values())
        
        # Check intersection
        matched = ext_rgbs.intersection(orig_rgbs)
        unmatched = ext_rgbs - orig_rgbs
        
        total_extracted += len(ext_rgbs)
        total_matched += len(matched)
        total_unmatched += len(unmatched)
        
        print(f"[{basename}] 提取颜色数量(Extracted): {len(ext_rgbs)}, 匹配原图(Matched): {len(matched)}, 未匹配(Unmatched): {len(unmatched)}")
        if len(unmatched) > 0:
            print(f"  --> 发现未匹配的异常颜色(Unmatched RGBs): {list(unmatched)[:5]}")
            
    print("-" * 50)
    print(f"总计提取面颜色数: {total_extracted}")
    print(f"总计完美匹配数:   {total_matched}")
    print(f"总计未匹配误差数: {total_unmatched}")
    
    if total_unmatched == 0 and total_extracted > 0:
        print("\n验证通过：所有从图片提取的像素RGB都完美包含在原本JSON保存的面染色RGB列表中！")
    else:
        print("\n验证失败：部分提取出的颜色与原STEP面颜色不一致（可能原因：渲染引擎抗锯齿、光照未完全关闭或色彩空间转换误差）。")

if __name__ == "__main__":
    verify_extracted_colors()