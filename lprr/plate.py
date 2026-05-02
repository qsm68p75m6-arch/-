# coding=utf-8
"""
车牌识别模块（LPRNet 推理）

设计原则：
  - 字符集在模块导入时静态确定（CHARS = CHARS_FULL，77类），不在运行时切换
  - 首次启动时若只有旧的 68 类权重，自动扩展并保存为 Final_LPRNet_model_full.pth
  - 后续直接加载 Final_LPRNet_model_full.pth，启动速度快，RK/RKNN 部署友好
  - 模型单例，全局只初始化一次

改进说明：
  1. 模型改为单例懒加载，避免每帧重复初始化
  2. 修复 Windows 路径硬编码，改用 os.path 动态定位
  3. GPU/CPU 自动选择，不再强制 cuda:0
  4. 输入合法性检查，裁剪区域为空时返回空结果
  5. 基于国标规则的后处理校验与修正
"""

import os
import numpy as np
import cv2
import torch

from lprr.LPRNet import build_lprnet, expand_output_layer, INPUT_W, INPUT_H
from lprr.chars_config import (
    CHARS,                  # 始终是 CHARS_FULL（77类）
    ORIGINAL_CLASS_NUM,     # 68
    FULL_CLASS_NUM,         # 77
    PROVINCE_CHARS,
    SPECIAL_CHARS,
    NEW_ENERGY_SUFFIX,
)

# ── 模型文件路径 ──────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_ORIGINAL = os.path.join(_THIS_DIR, 'Final_LPRNet_model.pth')
_MODEL_FULL     = os.path.join(_THIS_DIR, 'Final_LPRNet_model_full.pth')

# ── 单例 ──────────────────────────────────────────────────────────────────────
_lprnet = None
_device = None


def _get_lprnet():
    """
    懒加载 LPRNet，全局只初始化一次。

    加载策略（固定使用 CHARS_FULL / 77 类）：
      1. 若 Final_LPRNet_model_full.pth 存在 → 直接加载
      2. 若只有 Final_LPRNet_model.pth（68类）→ 自动扩展输出层并保存为 _full.pth
         下次启动直接走第 1 步，不再重复扩展
    """
    global _lprnet, _device
    if _lprnet is not None:
        return _lprnet, _device

    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if os.path.exists(_MODEL_FULL):
        # ── 直接加载已扩展的完整模型 ──────────────────────────────────────────
        _lprnet = build_lprnet(lpr_max_len=8, phase=False,
                               class_num=FULL_CLASS_NUM, dropout_rate=0.5)
        _lprnet.to(_device)
        state = torch.load(_MODEL_FULL, map_location=_device)
        # float16 权重自动转回 float32 加载（推理用 float32 更稳定）
        state = {k: v.float() if v.dtype == torch.float16 else v for k, v in state.items()}
        _lprnet.load_state_dict(state)
        _lprnet.eval()
        print(f'[LPRNet] 加载完整国标模型（{FULL_CLASS_NUM}类）: {_MODEL_FULL}')

    elif os.path.exists(_MODEL_ORIGINAL):
        # ── 首次运行：扩展旧模型并保存 ────────────────────────────────────────
        print(f'[LPRNet] 首次运行：将 {ORIGINAL_CLASS_NUM} 类模型扩展到 {FULL_CLASS_NUM} 类...')
        print(f'[LPRNet] 新增字符：港 澳 使 领 警 学 挂 试 练')
        _lprnet = expand_output_layer(
            old_model_path=_MODEL_ORIGINAL,
            new_model_path=_MODEL_FULL,   # 保存，下次直接加载
            device=_device,
        )
        print(f'[LPRNet] 扩展完成，已保存到: {_MODEL_FULL}')
        print(f'[LPRNet] 注意：新字符识别需微调训练，运行 python -m lprr.train_finetune 获得最佳效果')

    else:
        raise FileNotFoundError(
            f'未找到 LPRNet 权重文件。\n'
            f'请确认以下文件之一存在：\n'
            f'  {_MODEL_FULL}\n'
            f'  {_MODEL_ORIGINAL}'
        )

    print(f'[LPRNet] 就绪，设备: {_device}，字符集: {FULL_CLASS_NUM} 类')
    return _lprnet, _device


def _transform(img):
    """图像预处理：归一化 + CHW 转置。"""
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


def _ctc_decode(preb):
    """
    CTC 贪心解码：去除重复字符和空白符（'-'，索引为 len(CHARS)-1）。

    Args:
        preb: numpy array，shape [class_num, seq_len]

    Returns:
        list of int，字符索引列表
    """
    blank_idx = len(CHARS) - 1
    preb_label = [int(np.argmax(preb[:, j])) for j in range(preb.shape[1])]

    result = []
    prev = preb_label[0]
    if prev != blank_idx:
        result.append(prev)

    for c in preb_label[1:]:
        if c == prev or c == blank_idx:
            if c == blank_idx:
                prev = c
            continue
        result.append(c)
        prev = c

    return result


def _postprocess(char_indices):
    """
    基于国标规则对识别结果做后处理校验与修正。

    规则：
      - 首位必须是合法汉字（省份简称或特种汉字）
      - 普通民用车牌第二位字母不应为 I 或 O（纠正为 1 或 0）
      - 新能源车牌末位为 D 或 F，共 8 位
    """
    if not char_indices:
        return char_indices, 'unknown'

    char_list = [CHARS[i] for i in char_indices]
    plate_type = 'normal'

    if char_list and char_list[0] in SPECIAL_CHARS:
        plate_type = 'special'
    elif len(char_list) >= 8 and char_list[-1] in NEW_ENERGY_SUFFIX:
        plate_type = 'new_energy'

    # 普通民用牌：第二位 I→1，O→0
    if plate_type == 'normal' and len(char_list) >= 2:
        if char_list[1] == 'I' and '1' in CHARS:
            char_indices[1] = CHARS.index('1')
        elif char_list[1] == 'O' and '0' in CHARS:
            char_indices[1] = CHARS.index('0')

    return char_indices, plate_type


def de_lpr(coord, im0):
    """
    从原图中裁剪车牌区域并进行字符识别。

    Args:
        coord: 车牌 bounding box，格式 [x1, y1, x2, y2]（tensor 或 list）
        im0:   原始 BGR 图像（numpy array）

    Returns:
        plat_num:  numpy array，shape [1, n_chars]，字符索引（对应 CHARS）
        plate_img: numpy array，裁剪并缩放后的车牌图像 (94×24)
    """
    x1, y1, x2, y2 = int(coord[0]), int(coord[1]), int(coord[2]), int(coord[3])

    # 边界保护
    h, w = im0.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        print('[LPRNet] 警告：车牌裁剪区域无效，跳过识别')
        return np.array([[]]), np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)

    crop = im0[y1:y2, x1:x2]
    plate_img = cv2.resize(crop, (INPUT_W, INPUT_H))  # 94×48

    im = _transform(plate_img)
    ims = torch.tensor(np.array([im]))  # [1, 3, 24, 94]

    lprnet, device = _get_lprnet()

    with torch.no_grad():
        prebs = lprnet(ims.to(device))  # [1, class_num, seq_len]
        prebs = prebs.cpu().numpy()

    preb_labels = []
    for i in range(prebs.shape[0]):
        indices = _ctc_decode(prebs[i])
        indices, _ = _postprocess(indices)
        preb_labels.append(indices)

    if preb_labels and all(len(r) == len(preb_labels[0]) for r in preb_labels):
        plat_num = np.array(preb_labels)
    else:
        plat_num = np.empty(len(preb_labels), dtype=object)
        for i, row in enumerate(preb_labels):
            plat_num[i] = row

    return plat_num, plate_img


def decode_plate_number(plat_num):
    """
    将字符索引数组转换为车牌号码字符串。

    Args:
        plat_num: de_lpr 返回的 numpy array

    Returns:
        str，车牌号码，如 '京A12345' 或 '港A12345'
    """
    if plat_num is None or plat_num.size == 0:
        return ''
    try:
        indices = plat_num[0]
        return ''.join(CHARS[i] for i in indices)
    except (IndexError, TypeError):
        return ''


def reset_lprnet():
    """重置单例（用于重新加载权重，如微调后切换模型）。"""
    global _lprnet, _device
    _lprnet = None
    _device = None
    print('[LPRNet] 模型已重置，下次调用将重新加载。')
