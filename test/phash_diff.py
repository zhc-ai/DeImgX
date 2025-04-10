from PIL import Image
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 路径配置
path_a = "test/pre.png"
path_b = "test/overlay.png"
output_path = "test/high_fidelity_overlay.png"

# 读取图像并转换为 RGBA 模式
img_a = Image.open(path_a).convert("RGBA")
img_b = Image.open(path_b).convert("RGBA")

# 尺寸对齐（如果 a 与 b 尺寸不同，则缩放 a 以匹配 b）
if img_a.size != img_b.size:
    img_a = img_a.resize(img_b.size, resample=Image.BICUBIC)

# 高保真合成 a 在上，b 为底
composite = Image.alpha_composite(img_b, img_a)

# 保存为 PNG（无损）
composite.save(output_path)
print(f"✅ 合成图已保存为：{output_path}")

# ==== pHash 计算 ====
phash_b = imagehash.phash(img_b.convert("RGB"))
phash_composite = imagehash.phash(composite.convert("RGB"))
phash_diff = phash_b - phash_composite

print("原图 pHash：     ", phash_b)
print("合成图 pHash：   ", phash_composite)
print("pHash 差异位数：", phash_diff)

# ==== SSIM 计算 ====
# 转为灰度数组
gray_b = np.array(img_b.convert("L"))
gray_comp = np.array(composite.convert("L"))

# 结构相似度计算
ssim_score = ssim(gray_b, gray_comp)
print("SSIM（结构相似度）：", round(ssim_score, 4))
