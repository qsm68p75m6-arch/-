# coding=utf-8
"""
LPRNet 微调训练脚本
目标：在原始预训练权重基础上，通过迁移学习让模型识别完整国标字符集
     （新增：港/澳/使/领/警/学/挂/试/练）

训练策略：
  1. 用 expand_output_layer() 将旧权重（68类）迁移到新字符集（77类）
  2. 冻结 backbone 前几层（已学到的底层特征），只训练后几层和 container
  3. 用合成数据微调，学习新字符的视觉特征
  4. 训练完成后保存为 Final_LPRNet_model_full.pth

使用方法：
  # 第一步：生成合成数据（约需 2-5 分钟）
  python -m lprr.generate_plates --output_dir lprr/synthetic_data --num_samples 50000

  # 第二步：微调训练（GPU 约需 30-60 分钟，CPU 约需数小时）
  python -m lprr.train_finetune --data_dir lprr/synthetic_data --epochs 30

  # 训练完成后，chars_config.py 中的 CHARS 会自动切换到 CHARS_FULL
  # 重启应用即可使用完整国标字符集
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 确保可以从项目根目录运行
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from lprr.LPRNet import build_lprnet, expand_output_layer, INPUT_W, INPUT_H
from lprr.chars_config import (
    CHARS_FULL, FULL_CLASS_NUM, ORIGINAL_CLASS_NUM,
    CHAR2IDX_FULL,
)

_MODEL_PATH_ORIGINAL = os.path.join(_THIS_DIR, 'Final_LPRNet_model.pth')
_MODEL_PATH_FINETUNED = os.path.join(_THIS_DIR, 'Final_LPRNet_model_full.pth')
_CHARS_CONFIG_PATH = os.path.join(_THIS_DIR, 'chars_config.py')


# ── 数据集 ────────────────────────────────────────────────────────────────────

class PlateDataset(Dataset):
    """
    车牌图像数据集。
    标签文件格式：每行 "图像文件名 车牌号码"，如 "train_000001.jpg 京A12345"
    """

    def __init__(self, data_dir, split='train', chars=None):
        """
        Args:
            data_dir: 数据集根目录（包含 train/ 和 val/ 子目录）
            split:    'train' 或 'val'
            chars:    字符集列表，None 时使用 CHARS_FULL
        """
        self.chars = chars or CHARS_FULL
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.img_dir = os.path.join(data_dir, split, 'images')
        label_file = os.path.join(data_dir, split, 'labels.txt')

        self.samples = []  # list of (img_path, label_indices)
        skipped = 0

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                filename, plate_number = parts
                img_path = os.path.join(self.img_dir, filename)
                if not os.path.exists(img_path):
                    skipped += 1
                    continue
                indices = self._encode(plate_number)
                if indices is None:
                    skipped += 1
                    continue
                self.samples.append((img_path, indices))

        print(f'[Dataset] {split}: {len(self.samples)} 样本，跳过 {skipped} 个')

    def _encode(self, plate_number):
        """将车牌号码编码为字符索引列表，含未知字符返回 None。"""
        indices = []
        for ch in plate_number:
            if ch not in self.char2idx:
                return None
            indices.append(self.char2idx[ch])
        return indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_indices = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            # 读取失败时返回空白图像
            img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)

        # 确保尺寸正确
        img = cv2.resize(img, (INPUT_W, INPUT_H))  # 94×48

        # 归一化 + CHW
        img = img.astype('float32') / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img)

        return img_tensor, label_indices


def collate_fn(batch):
    """自定义 collate，处理不等长标签序列。"""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(labels)


# ── CTC Loss ──────────────────────────────────────────────────────────────────

class CTCLoss(nn.Module):
    """CTC 损失函数封装。"""

    def __init__(self, blank_idx):
        super(CTCLoss, self).__init__()
        self.blank_idx = blank_idx
        self.ctc = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)

    def forward(self, logits, labels):
        """
        Args:
            logits: [batch, class_num, seq_len]
            labels: list of list of int

        Returns:
            loss scalar
        """
        # CTCLoss 需要 [seq_len, batch, class_num]
        log_probs = torch.log_softmax(logits, dim=1)
        log_probs = log_probs.permute(2, 0, 1)  # [seq_len, batch, class_num]

        batch_size = logits.size(0)
        seq_len = logits.size(2)

        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

        # 将标签展平
        targets = torch.tensor(
            [idx for label in labels for idx in label],
            dtype=torch.long
        )

        return self.ctc(log_probs, targets, input_lengths, target_lengths)


# ── 训练 ──────────────────────────────────────────────────────────────────────

def freeze_backbone_layers(model, freeze_until_layer=14):
    """
    冻结 backbone 前 N 层（保留底层特征提取能力，只训练高层）。
    默认冻结到第 14 层（MaxPool3d），只训练后面的 Conv2d 和 container。
    """
    frozen = 0
    for i, layer in enumerate(model.backbone.children()):
        if i <= freeze_until_layer:
            for param in layer.parameters():
                param.requires_grad = False
            frozen += 1
        else:
            for param in layer.parameters():
                param.requires_grad = True

    # container 始终训练
    for param in model.container.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'[Train] 冻结前 {frozen} 层，可训练参数: {trainable:,} / {total:,}')


def evaluate(model, val_loader, criterion, device, chars):
    """在验证集上评估模型，返回 loss 和字符准确率。"""
    model.eval()
    total_loss = 0.0
    correct_chars = 0
    total_chars = 0
    correct_plates = 0
    total_plates = 0

    blank_idx = len(chars) - 1

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 贪心解码
            preds = logits.cpu().numpy()
            for i, pred in enumerate(preds):
                # pred shape: [class_num, seq_len]
                pred_label = []
                prev = -1
                for j in range(pred.shape[1]):
                    c = int(np.argmax(pred[:, j]))
                    if c != prev and c != blank_idx:
                        pred_label.append(c)
                    prev = c

                gt = labels[i]
                total_plates += 1
                if pred_label == gt:
                    correct_plates += 1

                # 字符级准确率（按最短对齐）
                min_len = min(len(pred_label), len(gt))
                for k in range(min_len):
                    total_chars += 1
                    if pred_label[k] == gt[k]:
                        correct_chars += 1
                total_chars += abs(len(pred_label) - len(gt))

    n = len(val_loader)
    avg_loss = total_loss / n if n > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    plate_acc = correct_plates / total_plates if total_plates > 0 else 0
    return avg_loss, char_acc, plate_acc


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[Train] 使用设备: {device}')
    print(f'[Train] 字符集大小: {FULL_CLASS_NUM} 类')

    # ── 加载并扩展模型 ────────────────────────────────────────────────────────
    if os.path.exists(_MODEL_PATH_FINETUNED) and not args.from_scratch:
        print(f'[Train] 从已有微调模型继续训练: {_MODEL_PATH_FINETUNED}')
        model = build_lprnet(lpr_max_len=8, phase='train',
                             class_num=FULL_CLASS_NUM, dropout_rate=0.5)
        model.to(device)
        state = torch.load(_MODEL_PATH_FINETUNED, map_location=device)
        model.load_state_dict(state)
    elif os.path.exists(_MODEL_PATH_ORIGINAL):
        print(f'[Train] 从原始模型迁移: {_MODEL_PATH_ORIGINAL}')
        model = expand_output_layer(
            old_model_path=_MODEL_PATH_ORIGINAL,
            new_model_path=None,
            device=device,
        )
        model.to(device)   # ← 修复：确保模型在正确设备上
        model.train()
    else:
        print('[Train] 未找到预训练权重，从随机初始化开始训练')
        model = build_lprnet(lpr_max_len=8, phase='train',
                             class_num=FULL_CLASS_NUM, dropout_rate=0.5)
        model.to(device)

    # ── 冻结策略 ──────────────────────────────────────────────────────────────
    if args.freeze_layers > 0:
        freeze_backbone_layers(model, freeze_until_layer=args.freeze_layers)

    # ── 数据集 ────────────────────────────────────────────────────────────────
    if not os.path.exists(args.data_dir):
        print(f'[Train] 数据目录不存在: {args.data_dir}')
        print('[Train] 请先运行: python -m lprr.generate_plates')
        return

    train_dataset = PlateDataset(args.data_dir, split='train', chars=CHARS_FULL)
    val_dataset   = PlateDataset(args.data_dir, split='val',   chars=CHARS_FULL)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── 损失函数和优化器 ──────────────────────────────────────────────────────
    blank_idx = len(CHARS_FULL) - 1
    criterion = CTCLoss(blank_idx=blank_idx)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_plate_acc = 0.0
    print(f'\n[Train] 开始训练，共 {args.epochs} 个 epoch')
    print(f'[Train] 批大小: {args.batch_size}，学习率: {args.lr}')
    print('-' * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg = total_loss / (batch_idx + 1)
                print(f'  Epoch {epoch}/{args.epochs} '
                      f'[{batch_idx+1}/{len(train_loader)}] '
                      f'loss={avg:.4f}', end='\r')

        scheduler.step()

        # 验证
        val_loss, char_acc, plate_acc = evaluate(
            model, val_loader, criterion, device, CHARS_FULL
        )
        elapsed = time.time() - t0
        avg_train_loss = total_loss / len(train_loader)

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'train_loss={avg_train_loss:.4f} | '
              f'val_loss={val_loss:.4f} | '
              f'char_acc={char_acc:.4f} | '
              f'plate_acc={plate_acc:.4f} | '
              f'lr={scheduler.get_last_lr()[0]:.6f} | '
              f'{elapsed:.1f}s')

        # 保存最优模型（float16 格式，约 396 KB，满足 500 KB 目标）
        if plate_acc > best_plate_acc:
            best_plate_acc = plate_acc
            # 转为 float16 保存，推理时自动转回 float32
            fp16_state = {k: v.half() for k, v in model.state_dict().items()}
            torch.save(fp16_state, _MODEL_PATH_FINETUNED)
            size_kb = os.path.getsize(_MODEL_PATH_FINETUNED) / 1024
            print(f'  ✓ 保存最优模型 float16 (plate_acc={plate_acc:.4f}, {size_kb:.0f} KB): {_MODEL_PATH_FINETUNED}')

    print(f'\n[Train] 训练完成！最优整牌准确率: {best_plate_acc:.4f}')
    print(f'[Train] 模型已保存到: {_MODEL_PATH_FINETUNED}')

    # ── 自动更新 chars_config.py 中的 CHARS 切换 ─────────────────────────────
    _update_chars_config()


def _update_chars_config():
    """
    训练完成后，自动将 chars_config.py 中的 CHARS = CHARS_ORIGINAL
    替换为 CHARS = CHARS_FULL，使应用自动使用完整字符集。
    """
    try:
        with open(_CHARS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            content = f.read()

        old_line = 'CHARS = CHARS_ORIGINAL'
        new_line = 'CHARS = CHARS_FULL'

        if old_line in content:
            content = content.replace(old_line, new_line)
            with open(_CHARS_CONFIG_PATH, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'[Train] ✓ 已自动更新 chars_config.py: CHARS = CHARS_FULL')
            print('[Train] 重启应用后将使用完整国标字符集（77类）')
        elif new_line in content:
            print('[Train] chars_config.py 已经是 CHARS = CHARS_FULL，无需更新')
        else:
            print('[Train] 警告：未能自动更新 chars_config.py，请手动将 CHARS = CHARS_ORIGINAL 改为 CHARS = CHARS_FULL')
    except Exception as e:
        print(f'[Train] 警告：更新 chars_config.py 失败: {e}')
        print('[Train] 请手动将 chars_config.py 中的 CHARS = CHARS_ORIGINAL 改为 CHARS = CHARS_FULL')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPRNet 微调训练（完整国标字符集）')
    parser.add_argument('--data_dir', type=str,
                        default='lprr/synthetic_data',
                        help='合成数据集目录（由 generate_plates.py 生成）')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数（建议 20-50）')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批大小（GPU 建议 128-256，CPU 建议 32）')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--freeze_layers', type=int, default=14,
                        help='冻结 backbone 前 N 层（0=不冻结，14=只训练后几层）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作进程数（Windows 建议设为 0）')
    parser.add_argument('--from_scratch', action='store_true',
                        help='忽略已有微调模型，从原始权重重新开始')
    args = parser.parse_args()

    train(args)
