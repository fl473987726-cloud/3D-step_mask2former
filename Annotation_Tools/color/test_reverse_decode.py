# -*- coding: utf-8 -*-
"""验证反向解码：从RGB找回face_id、类型、面积"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from color_encoder import FaceColorEncoder, TYPE_NAMES

MAPPING_PATH = r"E:\soft\code\Mask2former\results\color_mapping\test_random\mapping.json"

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

encoder = FaceColorEncoder.from_mapping(MAPPING_PATH)

# 从 extra_config 中获取原始 face 数据
faces_info = data["config"]["faces"]

print("反向解码验证:")
print(f"{'面ID':<6}{'原始类型':<8}{'原始面积':<10}{'解码类型':<8}{'解码面积':<10}{'匹配'}")
print("-" * 60)

correct = 0
for fid, info in faces_info.items():
    type_id_orig = info["type_id"]
    area_ratio_orig = info["area_ratio"]
    g = data["faces"][fid]["G"]
    b = data["faces"][fid]["B"]

    # 用原始类型+面积编码成 RGB，再解码
    r = encoder.encode(int(fid), type_id_orig, area_ratio_orig)[0]
    result = encoder.decode((r, g, b))

    type_id_decoded = result["type_id"]
    area_decoded = result["area_ratio"]
    face_id_decoded = result["face_id"]

    match = "OK" if type_id_decoded == type_id_orig and str(face_id_decoded) == fid else "FAIL"
    if match == "OK":
        correct += 1

    print(f"{fid:<6}{TYPE_NAMES[type_id_orig]:<8}{area_ratio_orig:<10.2%}"
          f"{TYPE_NAMES[type_id_decoded]:<8}{area_decoded:<10.2%}{match}")

print(f"\n正确率: {correct}/{len(faces_info)} ({correct/len(faces_info)*100:.0f}%)")
