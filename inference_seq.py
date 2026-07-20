import sys
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
from io import BytesIO

# ---------- ВКЛЮЧАЕМ СОВМЕСТИМОСТЬ С NUMPY ----------
sys.path.insert(0, '/usr/lib/python3/dist-packages')
os.environ['NPY_ARRAY_API_VERSION'] = '1'

# ---------- МОДЕЛЬ U‑NET (ОБЩАЯ ДЛЯ ВСЕХ) ----------
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

# ---------- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: ПРЕВРАЩАЕМ PIL В BASE64 ----------
def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# ---------- ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ ----------
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# ---------- ФУНКЦИЯ СЕГМЕНТАЦИИ (ВОЗВРАЩАЕТ ОРИГИНАЛ, МАСКУ, НАЛОЖЕНИЕ) ----------
def get_images(image_path, model_path, img_size=(256, 256), threshold=0.5):
    model, device = load_model(model_path)
    
    # Загружаем оригинал
    img_pil = Image.open(image_path).convert('L')
    original_size = img_pil.size
    
    # Оригинал в цвете (для отображения)
    original_rgb = img_pil.convert('RGB')
    
    # Подготовка для модели
    img_resized = img_pil.resize(img_size, Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        pred = model(img_tensor)
        mask = (pred > threshold).float().cpu().numpy().squeeze()

    # Маска в оригинальном размере
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize(original_size, Image.NEAREST)
    mask_resized_np = np.array(mask_resized, dtype=np.float32) / 255.0

    # Наложение (оригинал + красная маска)
    overlay = np.array(original_rgb)
    mask_bool = mask_resized_np > 0.5
    overlay[mask_bool] = (overlay[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)

    # Возвращаем три отдельные картинки в base64
    return {
        'original': pil_to_base64(original_rgb),
        'mask': pil_to_base64(mask_resized),
        'overlay': pil_to_base64(overlay_pil)
    }

# ---------- ОСНОВНАЯ ФУНКЦИЯ ----------
def predict(image_path):
    # ВРЕМЕННО: определяем тип опухоли по наличию файлов моделей
    model_files = {
        'glioma': 'glioma.pth',
        'meningioma': 'meningioma.pth',
        'pituitary': 'pituitary.pth'
    }
    
    detected_type = None
    for tumor_type, model_file in model_files.items():
        if os.path.exists(model_file):
            detected_type = tumor_type
            break
    
    if detected_type is None:
        raise FileNotFoundError("Не найдено ни одной модели сегментации! Загрузите glioma.pth, meningioma.pth или pituitary.pth")
    
    display_names = {
        'glioma': 'Глиома',
        'meningioma': 'Менингиома',
        'pituitary': 'Опухоль гипофиза'
    }
    
    tumor_descriptions = {
        'glioma': '🧠 Глиома — опухоль, которая развивается из глиальных клеток (клетки, поддерживающие нейроны). Чаще всего встречается в головном мозге. Может быть как доброкачественной, так и злокачественной. Требует наблюдения и лечения.',
        'meningioma': '🧠 Менингиома — опухоль, которая развивается из оболочек мозга (мозговых оболочек). В большинстве случаев доброкачественная, растёт медленно. Часто не требует срочного лечения, но нужен контроль.',
        'pituitary': '🧠 Опухоль гипофиза — новообразование в гипофизе (железе, которая регулирует гормоны). Чаще всего доброкачественная. Может влиять на гормональный фон и зрение. Требует наблюдения у эндокринолога.'
    }
    
    model_file = model_files[detected_type]
    
    # Делаем сегментацию (получаем три картинки)
    images = get_images(image_path, model_file)
    
    # Возвращаем результат с ТРЕМЯ картинками
    return {
        'class': display_names[detected_type],
        'description': tumor_descriptions[detected_type],
        'original': images['original'],
        'mask': images['mask'],
        'overlay': images['overlay']
    }

# ---------- ЗАПУСК ----------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write("Ошибка: укажите путь к изображению")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        result = predict(image_path)
        print(json.dumps(result))
    except Exception as e:
        sys.stderr.write(str(e))
        sys.exit(1)
