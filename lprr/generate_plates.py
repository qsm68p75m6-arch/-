# coding=utf-8
"""
合成车牌图像数据集生成器

用途：为微调 LPRNet 生成包含完整国标字符集的合成训练数据。
重点生成原始模型缺失的字符：港、澳、使、领、警、学、挂、试、练

生成策略：
  - 用 PIL 在标准车牌尺寸（440×140）上渲染文字，然后缩放到 LPRNet 输入尺寸（94×24）
  - 对每种车牌类型按国标格式随机生成号码
  - 增加随机噪声、亮度变化、轻微旋转等数据增强，提升泛化能力
  - 输出格式：图像文件 + 标签文件（每行：图像路径 车牌号码）

使用方法：
  python -m lprr.generate_plates --output_dir lprr/synthetic_data --num_samples 50000
"""

import os
import random
import argparse
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from lprr.chars_config import (
    CHARS_FULL, PROVINCE_CHARS, SPECIAL_CHARS,
    CHAR2IDX_FULL, NEW_ENERGY_SUFFIX,
)
from lprr.LPRNet import INPUT_W, INPUT_H

# ── 车牌颜色配置（BGR）────────────────────────────────────────────────────────
PLATE_COLORS = {
    'blue':       {'bg': (180, 100, 30),   'text': (255, 255, 255)},  # 蓝底白字（普通民用）
    'yellow':     {'bg': (0, 200, 230),    'text': (0, 0, 0)},        # 黄底黑字（货车/挂车）
    'green':      {'bg': (50, 160, 50),    'text': (255, 255, 255)},  # 绿底白字（新能源）
    'white':      {'bg': (240, 240, 240),  'text': (0, 0, 0)},        # 白底黑字（使/领/警）
    'black':      {'bg': (30, 30, 30),     'text': (255, 255, 255)},  # 黑底白字（港澳）
}

# ── 车牌类型 → 颜色映射 ────────────────────────────────────────────────────────
TYPE_COLOR = {
    'normal':     'blue',
    'new_energy': 'green',
    'truck':      'yellow',
    'hang':       'yellow',   # 挂车
    'police':     'white',    # 警察
    'embassy':    'white',    # 使/领馆
    'hk_mo':      'black',    # 港澳
    'school':     'yellow',   # 学车
    'test':       'white',    # 试验
    'train':      'yellow',   # 教练
}

# 普通民用车牌字母（不含 I/O）
NORMAL_LETTERS = [c for c in 'ABCDEFGHJKLMNPQRSTUVWXYZ']
# 完整字母（含 I/O，特种牌用）
ALL_LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DIGITS = list('0123456789')


def _random_normal_plate():
    """生成普通民用车牌号码，格式：省份 + 字母 + 5位字母数字"""
    province = random.choice(list(PROVINCE_CHARS))
    letter = random.choice(NORMAL_LETTERS)
    rest = ''.join(random.choices(NORMAL_LETTERS + DIGITS, k=5))
    return province + letter + rest, 'normal'


def _random_new_energy_plate():
    """生成新能源车牌，格式：省份 + 字母 + 5位（末位D或F）"""
    province = random.choice(list(PROVINCE_CHARS))
    letter = random.choice(NORMAL_LETTERS)
    mid = ''.join(random.choices(NORMAL_LETTERS + DIGITS, k=4))
    suffix = random.choice(['D', 'F'])
    return province + letter + mid + suffix, 'new_energy'


def _random_special_plate(first_char, plate_type):
    """生成特种车牌，格式：特种汉字 + 字母 + 5位字母数字"""
    letter = random.choice(ALL_LETTERS)
    rest = ''.join(random.choices(ALL_LETTERS + DIGITS, k=5))
    return first_char + letter + rest, plate_type


def _random_hk_mo_plate(first_char):
    """生成港澳车牌，格式：港/澳 + 字母 + 5位"""
    letter = random.choice(ALL_LETTERS)
    rest = ''.join(random.choices(ALL_LETTERS + DIGITS, k=5))
    return first_char + letter + rest, 'hk_mo'


def generate_plate_number():
    """
    随机生成一个合法的国标车牌号码。
    各类型按比例生成，重点增加特种车牌比例以平衡训练数据。
    """
    r = random.random()
    if r < 0.50:
        return _random_normal_plate()
    elif r < 0.62:
        return _random_new_energy_plate()
    elif r < 0.67:
        return _random_special_plate('警', 'police')
    elif r < 0.70:
        return _random_special_plate('学', 'school')
    elif r < 0.73:
        return _random_special_plate('挂', 'hang')
    elif r < 0.76:
        return _random_special_plate('使', 'embassy')
    elif r < 0.79:
        return _random_special_plate('领', 'embassy')
    elif r < 0.82:
        return _random_hk_mo_plate('港')
    elif r < 0.85:
        return _random_hk_mo_plate('澳')
    elif r < 0.88:
        return _random_special_plate('试', 'test')
    elif r < 0.91:
        return _random_special_plate('练', 'train')
    else:
        return _random_normal_plate()


def _find_font(size=60):
    """查找可用的中文字体。"""
    candidates = [
        # Windows
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simfang.ttf',
        # Linux
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        # macOS
        '/System/Library/Fonts/PingFang.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        # 项目目录
        'simsun.ttc',
        'Arial.ttf',
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                from PIL import ImageFont
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # 最后回退到默认字体（不支持中文，但至少不崩溃）
    from PIL import ImageFont
    return ImageFont.load_default()


def render_plate_image(plate_number, plate_type, width=INPUT_W, height=INPUT_H,
                       augment=True):
    """
    渲染车牌图像。

    Args:
        plate_number: 车牌号码字符串，如 '京A12345'
        plate_type:   车牌类型字符串
        width:        输出图像宽度（LPRNet 输入为 94）
        height:       输出图像高度（LPRNet 输入为 24）
        augment:      是否进行数据增强

    Returns:
        numpy array，BGR 格式，shape [height, width, 3]
    """
    color_key = TYPE_COLOR.get(plate_type, 'blue')
    colors = PLATE_COLORS[color_key]

    # 先在较大尺寸上渲染，再缩小（抗锯齿）
    scale = 4
    W, H = width * scale, height * scale

    img = Image.new('RGB', (W, H), colors['bg'])
    draw = ImageDraw.Draw(img)

    font_size = int(H * 0.75)
    font = _find_font(font_size)

    # 计算文字总宽度，居中绘制
    char_w = W // (len(plate_number) + 1)
    x_start = (W - char_w * len(plate_number)) // 2

    for i, ch in enumerate(plate_number):
        x = x_start + i * char_w
        y = (H - font_size) // 2
        draw.text((x, y), ch, fill=colors['text'], font=font)

    # 缩放到目标尺寸
    img = img.resize((width, height), Image.LANCZOS)
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if augment:
        img_bgr = _augment(img_bgr)

    return img_bgr


def _augment(img):
    """数据增强：噪声、亮度、对比度、轻微旋转、模糊。"""
    # 随机亮度
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.4)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 随机高斯噪声
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 随机模糊
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # 随机轻微旋转（±5度）
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 随机透视变换（模拟拍摄角度）
    if random.random() < 0.2:
        h, w = img.shape[:2]
        margin = int(h * 0.1)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), random.randint(0, margin)],
            [w - random.randint(0, margin), h - random.randint(0, margin)],
            [random.randint(0, margin), h - random.randint(0, margin)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img


def plate_to_label(plate_number, chars=None):
    """
    将车牌号码字符串转换为字符索引列表。

    Args:
        plate_number: 车牌号码字符串
        chars:        字符集列表，None 时使用 CHARS_FULL

    Returns:
        list of int，字符索引列表；若有字符不在字符集中返回 None
    """
    if chars is None:
        chars = CHARS_FULL
    char2idx = {c: i for i, c in enumerate(chars)}
    indices = []
    for ch in plate_number:
        if ch not in char2idx:
            return None  # 含有字符集外的字符，跳过
        indices.append(char2idx[ch])
    return indices


def generate_dataset(output_dir, num_samples=50000, val_ratio=0.1):
    """
    生成合成车牌数据集。

    Args:
        output_dir:  输出目录
        num_samples: 总样本数
        val_ratio:   验证集比例

    输出结构：
        output_dir/
          train/
            images/  *.jpg
            labels.txt  （每行：图像文件名 车牌号码）
          val/
            images/  *.jpg
            labels.txt
    """
    train_dir = os.path.join(output_dir, 'train', 'images')
    val_dir   = os.path.join(output_dir, 'val',   'images')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    train_labels = []
    val_labels   = []

    n_val   = int(num_samples * val_ratio)
    n_train = num_samples - n_val

    print(f'生成训练集: {n_train} 张，验证集: {n_val} 张')
    print(f'输出目录: {output_dir}')

    for i in range(num_samples):
        is_val = (i < n_val)
        split  = 'val' if is_val else 'train'
        img_dir = val_dir if is_val else train_dir

        plate_number, plate_type = generate_plate_number()
        label = plate_to_label(plate_number)
        if label is None:
            continue  # 跳过含未知字符的车牌

        img = render_plate_image(plate_number, plate_type, augment=not is_val)

        filename = f'{split}_{i:06d}.jpg'
        filepath = os.path.join(img_dir, filename)
        cv2.imwrite(filepath, img)

        entry = f'{filename} {plate_number}\n'
        if is_val:
            val_labels.append(entry)
        else:
            train_labels.append(entry)

        if (i + 1) % 5000 == 0:
            print(f'  已生成 {i + 1}/{num_samples}...')

    # 写标签文件
    with open(os.path.join(output_dir, 'train', 'labels.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    with open(os.path.join(output_dir, 'val', 'labels.txt'), 'w', encoding='utf-8') as f:
        f.writelines(val_labels)

    print(f'数据集生成完成！训练: {len(train_labels)} 张，验证: {len(val_labels)} 张')
    return len(train_labels), len(val_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='合成车牌数据集生成器')
    parser.add_argument('--output_dir', type=str,
                        default='lprr/synthetic_data',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='总样本数（建议 ≥ 50000）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.num_samples, args.val_ratio)
