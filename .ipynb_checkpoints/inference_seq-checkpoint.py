import sys
# Заставляем Python искать numpy в системной папке (важно!)
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import os
# Включаем совместимость с NumPy 1.x
os.environ['NPY_ARRAY_API_VERSION'] = '1'

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
import base64

# ---------- Модель U-Net ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature*2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx](x)
        return torch.sigmoid(self.final_conv(x))

def plot_to_base64(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def predict_and_visualize(image_path, model_path="best_unet.pth", img_size=(256, 256), threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_pil = Image.open(image_path).convert('L')
    original_size = img_pil.size

    img_resized = img_pil.resize(img_size, Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (pred > threshold).float().cpu().numpy().squeeze()

    # Создаём три графика
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_pil, cmap='gray')
    axes[0].set_title('Оригинал')
    axes[0].axis('off')

    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Предсказанная маска')
    axes[1].axis('off')

    mask_resized = Image.fromarray((pred_mask * 255).astype(np.uint8))
    mask_resized = mask_resized.resize(original_size, Image.NEAREST)
    mask_resized = np.array(mask_resized, dtype=np.float32) / 255.0

    overlay = np.array(img_pil.convert('RGB'))
    mask_bool = mask_resized > 0.5
    overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

    axes[2].imshow(overlay)
    axes[2].set_title('Наложение маски')
    axes[2].axis('off')

    plt.tight_layout()
    result_base64 = plot_to_base64(plt)
    plt.close(fig)
    return result_base64

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Ошибка: укажите путь к изображению")
        sys.exit(1)
    image_path = sys.argv[1]
    try:
        result = predict_and_visualize(image_path)
        print(result)
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)
