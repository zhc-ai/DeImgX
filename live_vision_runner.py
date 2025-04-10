import sys
import time
import av
import cv2  # ✅ 图像处理库 OpenCV
import numpy as np
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
import clip  # ✅ OpenAI 的图文对齐模型 CLIP
import imagehash  # ✅ 感知哈希
from PIL import Image

# ========== 配置参数 ==========
VIDEO_PATH = "sample.mp4"  # 输入视频路径
VIDEO_WIDTH = 1080          # 视频原始宽度
VIDEO_HEIGHT = 1920         # 视频原始高度
PREVIEW_WIDTH = 360         # 预览窗口宽度（等比例缩放）
PREVIEW_HEIGHT = int(PREVIEW_WIDTH * VIDEO_HEIGHT / VIDEO_WIDTH)
FRAME_RATE = 30             # 预览帧率
ENABLE_CLIP_ATTACK = False      # ✅ 开关：是否启用 CLIP 对抗扰动
ENABLE_CLIP_SIMILARITY = True   # ✅ 新增开关：是否计算混淆前后在 CLIP 嵌入空间的相似度
ENABLE_SSIM_FPS_LOG = True      # ✅ 开关：是否打印 SSIM 和 FPS
ENABLE_PHASH_CHECK = True       # ✅ 开关：是否启用感知哈希检测

# ========== 初始化设备和模型 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\U0001F4BB 使用设备: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

# ========== CLIP 对抗扰动函数 ==========
def clip_adversarial(image_tensor, original_embedding, steps=5, epsilon=0.03):
    adv_tensor = image_tensor.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([adv_tensor], lr=1e-2)

    text = clip.tokenize(["photo"]).to(device)
    text_features = model.encode_text(text).detach()

    for _ in range(steps):
        adv_clamped = adv_tensor.clamp(0, 1)
        adv_input = F.interpolate(adv_clamped, size=(224, 224), mode="bilinear", align_corners=False)
        adv_input.requires_grad_(True)

        image_features = model.encode_image(adv_input)
        logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
        loss = torch.cosine_similarity(image_features, original_embedding, dim=-1).mean()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        adv_tensor.data = adv_tensor.data.clamp(0, 1)

    return adv_tensor.detach()

# ========== 图像频域扰动函数 ==========
def spatial_obfuscation(tensor):
    fft = torch.fft.fft2(tensor)
    fft_shift = torch.fft.fftshift(fft)
    _, _, h, w = fft_shift.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=tensor.device), torch.arange(w, device=tensor.device), indexing='ij')
    center_h, center_w = h // 2, w // 2
    radius = 30
    mask = ((yy - center_h) ** 2 + (xx - center_w) ** 2 >= radius ** 2).float().to(tensor.device)
    mask = mask.unsqueeze(0).unsqueeze(0)
    noise = (torch.rand_like(fft_shift.real) - 0.5) * 0.6
    fft_shift += (noise + 1j * noise) * mask
    result = torch.fft.ifft2(torch.fft.ifftshift(fft_shift)).real
    return torch.clamp(result, 0, 1)

# ========== SSIM 计算函数 ==========
def compute_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(img1, 11, 1, 5)
    mu2 = F.avg_pool2d(img2, 11, 1, 5)
    sigma1 = F.avg_pool2d(img1 ** 2, 11, 1, 5) - mu1 ** 2
    sigma2 = F.avg_pool2d(img2 ** 2, 11, 1, 5) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 11, 1, 5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim_map.mean().item()

# ========== pHash 相似度计算 ==========
def compute_phash_sim(img1, img2):
    pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    hash1 = imagehash.phash(pil1)
    hash2 = imagehash.phash(pil2)
    return 1 - (hash1 - hash2) / len(hash1.hash) ** 2

# ========== OpenGL 预览窗口类 ==========
class GLPreview(QGLWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.texture_id = None
        self.frame_data = None

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        glViewport(0, 0, self.width(), self.height())
        self.texture_id = glGenTextures(1)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def update_frame(self, rgb_frame):
        self.frame_data = rgb_frame
        self.update()

    def paintGL(self):
        if self.frame_data is None:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, PREVIEW_WIDTH, PREVIEW_HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     self.frame_data.shape[1], self.frame_data.shape[0], 0,
                     GL_RGB, GL_UNSIGNED_BYTE, self.frame_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, PREVIEW_HEIGHT)
        glTexCoord2f(1, 1); glVertex2f(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        glTexCoord2f(1, 0); glVertex2f(PREVIEW_WIDTH, 0)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glEnd()

        glDisable(GL_TEXTURE_2D)

# ========== 主窗口类 ==========
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU 混淆预览 - PyOpenGL + PyAV")
        self.preview = GLPreview()
        self.setCentralWidget(self.preview)

        self.container = av.open(VIDEO_PATH)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_frame)
        self.timer.start(int(1000 / FRAME_RATE))

        self.frame_iter = self.container.decode(self.stream)
        self.last_time = time.time()
        self.frame_count = 0
        self.orig_embed = None

    def read_frame(self):
        try:
            frame = next(self.frame_iter)
            img = frame.to_ndarray(format="rgb24")

            h, w = img.shape[:2]
            scale = min(PREVIEW_WIDTH / w, PREVIEW_HEIGHT / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h))
            padded = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
            x_offset = (PREVIEW_WIDTH - new_w) // 2
            y_offset = (PREVIEW_HEIGHT - new_h) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
            img = padded

            tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            perturbed = spatial_obfuscation(tensor)

            if ENABLE_CLIP_ATTACK:
                resized_tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
                resized_tensor.requires_grad_(True)
                self.orig_embed = model.encode_image(resized_tensor).detach()
                adv_tensor = clip_adversarial(perturbed.clone(), self.orig_embed)
                output_np = (adv_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                output_np = (perturbed.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            if ENABLE_SSIM_FPS_LOG:
                ssim = compute_ssim(tensor, perturbed)
                self.frame_count += 1
                now = time.time()
                if now - self.last_time >= 1:
                    fps = self.frame_count / (now - self.last_time)
                    log = f"[实时] FPS: {fps:.2f} | SSIM: {ssim:.4f}"

                    if ENABLE_PHASH_CHECK:
                        phash_sim = compute_phash_sim(img, output_np)
                        log += f" | pHash相似度: {phash_sim:.4f}"

                    if ENABLE_CLIP_SIMILARITY and self.orig_embed is not None:
                        clip_out = F.interpolate(torch.from_numpy(output_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0,
                                                size=(224, 224), mode="bilinear", align_corners=False)
                        embed = model.encode_image(clip_out)
                        cos_sim = F.cosine_similarity(embed, self.orig_embed, dim=-1).item()
                        log += f" | CLIP相似度: {cos_sim:.4f}"

                    print(log)
                    self.last_time = now
                    self.frame_count = 0

            self.preview.update_frame(output_np)

        except StopIteration:
            self.frame_iter = self.container.decode(self.stream)

# ========== 启动应用 ==========
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())