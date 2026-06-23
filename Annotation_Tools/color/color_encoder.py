# -*- coding: utf-8 -*-
"""面颜色编码器

编码规则：
- R 通道：面类型 + 面积
  R = TYPE_R_BASE[type_id] + round(area_ratio * (TYPE_GAP - 1))
- G/B 通道：face_id 自适应二维网格编码
  K = ceil(sqrt(num_faces)), 每个 face_id 映射到 K x K 网格中的一个点
"""
import json
import math
import random

TYPE_GAP = 51
TYPE_R_BASE = {
    0: 0,    # Plane
    1: 51,   # Cylinder
    2: 102,  # Cone
    3: 153,  # Sphere
    4: 204,  # Other
}
TYPE_NAMES = {
    0: "Plane",
    1: "Cylinder",
    2: "Cone",
    3: "Sphere",
    4: "Other",
}
NAME_TO_TYPE_ID = {v: k for k, v in TYPE_NAMES.items()}


class FaceColorEncoder:
    def __init__(self, num_faces, shuffle=False, seed=42):
        if num_faces <= 0:
            raise ValueError("num_faces must be positive")
        self.num_faces = int(num_faces)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.K = math.ceil(math.sqrt(self.num_faces))
        if self.K > 255:
            raise ValueError(f"K={self.K} too large, GB grid supports up to 255x255 faces")
        self.gb_mapping = self._build_gb_mapping()
        self.reverse_gb_mapping = {tuple(v): int(k) for k, v in self.gb_mapping.items()}

    def _build_gb_mapping(self):
        ids = list(range(1, self.num_faces + 1))
        slots = list(range(self.K * self.K))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(slots)
        step = 254 / max(self.K - 1, 1)
        mapping = {}
        for face_id, slot in zip(ids, slots):
            row = slot // self.K
            col = slot % self.K
            g = int(round(row * step))
            b = int(round(col * step))
            g = max(0, min(254, g))
            b = max(0, min(254, b))
            mapping[face_id] = (g, b)
        return mapping

    def encode(self, face_id, type_id, area_ratio):
        face_id = int(face_id)
        type_id = int(type_id)
        if face_id not in self.gb_mapping:
            raise KeyError(f"face_id {face_id} not in encoder mapping")
        if type_id not in TYPE_R_BASE:
            type_id = 4
        area_ratio = max(0.0, min(1.0, float(area_ratio)))
        area_offset = int(round(area_ratio * (TYPE_GAP - 1)))
        r = TYPE_R_BASE[type_id] + area_offset
        g, b = self.gb_mapping[face_id]
        return int(r), int(g), int(b)

    def decode(self, rgb):
        r, g, b = [int(x) for x in rgb]
        type_id = min(TYPE_R_BASE.keys(), key=lambda tid: abs(r - TYPE_R_BASE[tid]))
        for tid, base in TYPE_R_BASE.items():
            if base <= r < base + TYPE_GAP:
                type_id = tid
                break
        area_offset = max(0, min(TYPE_GAP - 1, r - TYPE_R_BASE[type_id]))
        area_ratio = area_offset / (TYPE_GAP - 1)
        face_id = self.reverse_gb_mapping.get((g, b))
        return {
            "face_id": face_id,
            "type_id": type_id,
            "type_name": TYPE_NAMES.get(type_id, "Other"),
            "area_ratio": area_ratio,
            "R": r,
            "G": g,
            "B": b,
        }

    def save_mapping(self, path, extra_config=None):
        data = {
            "encoding_rule": {
                "R": "type base + area offset; TYPE_GAP=51",
                "G_B": "face_id adaptive grid mapping",
                "TYPE_R_BASE": TYPE_R_BASE,
                "TYPE_NAMES": TYPE_NAMES,
                "background_rgb": [255, 255, 255],
            },
            "config": {
                "num_faces": self.num_faces,
                "K": self.K,
                "shuffle": self.shuffle,
                "seed": self.seed,
            },
            "faces": {
                str(fid): {"G": int(g), "B": int(b)}
                for fid, (g, b) in self.gb_mapping.items()
            },
        }
        if extra_config:
            data["config"].update(extra_config)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_mapping(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data["config"]
        obj = cls(cfg["num_faces"], cfg.get("shuffle", False), cfg.get("seed", 42))
        obj.K = cfg.get("K", obj.K)
        obj.gb_mapping = {
            int(fid): (int(info["G"]), int(info["B"]))
            for fid, info in data["faces"].items()
        }
        obj.reverse_gb_mapping = {tuple(v): int(k) for k, v in obj.gb_mapping.items()}
        return obj
