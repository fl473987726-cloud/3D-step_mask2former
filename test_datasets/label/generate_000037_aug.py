import cv2
import numpy as np
import os

src = r'e:\aaaa-WUT\lw\ASCCAD\test_step\label\datatest01_0527\images\000037.png'
out = r'e:\aaaa-WUT\lw\ASCCAD\test_step\label\datatest01_0527\images\000037_aug'
os.makedirs(out, exist_ok=True)

img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
if img is None:
    raise RuntimeError('无法读取图片')

h, w = img.shape[:2]
center = (w / 2.0, h / 2.0)

results = [
    ('000037_flip_lr.png', cv2.flip(img, 1)),
    ('000037_flip_ud.png', cv2.flip(img, 0)),
]

for angle in [45, 90, 135, 180]:
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    results.append((f'000037_rot_{angle}.png', rotated))

for name, image in results:
    path = os.path.join(out, name)
    cv2.imencode('.png', image)[1].tofile(path)

print(f'已生成 {len(results)} 张增强图片到: {out}')
for name, _ in results:
    print(name)
