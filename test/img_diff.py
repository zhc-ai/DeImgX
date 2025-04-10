import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# 加载 CLIP 模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载两张图片（替换为你自己的路径）
img1 = Image.open("test/pre.jpg").convert("RGB")
img2 = Image.open("test/after.jpg").convert("RGB")

# 预处理图片
inputs = processor(images=[img1, img2], return_tensors="pt", padding=True)

# 获取图片特征
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# 归一化特征
image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

# 计算余弦相似度
similarity = torch.nn.functional.cosine_similarity(image_features[0], image_features[1], dim=0)

print(f"CLIP 相似度（0-1之间，越高越相似）: {similarity.item():.4f}")
