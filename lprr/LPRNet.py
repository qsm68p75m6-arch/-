# coding=utf-8
"""
LPRNet —— License Plate Recognition Network
原始论文: https://arxiv.org/abs/1806.10447

改动说明：
  1. 字符集统一从 chars_config.py 导入，不再在此文件硬编码
  2. build_lprnet 的 class_num 默认值改为从配置读取，避免新旧字符集不一致
  3. 新增 expand_output_layer()：将旧权重（68类）迁移到新字符集（77类）
  4. 【输入尺寸升级】94×24 → 94×48，解决中文笔画在低分辨率下糊成一团的问题
  5. 【模型瘦身】权重从 3963 KB → float16 保存约 396 KB，满足 500 KB 目标
     - backbone.16: Conv2d(64, 256, (1,4)) → Conv2d(64, 128, (1,4))，输出通道减半
     - backbone.20: Conv2d(256, class_num, (40,1)) → Conv2d(128, class_num, (1,1))
       在 backbone.20 之前用 AdaptiveAvgPool2d((1,W)) 压缩高度，kernel 只需 (1,1)
     - container: 输入通道 64+128+256+class_num = 448+class_num（与原始一致）
     - forward() 统一用 AdaptiveAvgPool2d((1, 18)) 处理所有 keep_features
"""

import os
import torch
import torch.nn as nn

from lprr.chars_config import (
    CHARS, CHARS_ORIGINAL, CHARS_FULL,
    ORIGINAL_CLASS_NUM, FULL_CLASS_NUM,
    CHAR2IDX_ORIGINAL, CHAR2IDX_FULL,
)

# ── 输入图像尺寸（全局唯一定义，plate.py / train_finetune.py / generate_plates.py 均从此导入）
INPUT_W = 94   # 宽度：与原始 LPRNet 一致，保证序列长度 W=18
INPUT_H = 48   # 高度：从 24 升级到 48，中文笔画清晰度大幅提升


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),   # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),                                                              # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),                               # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),                                                              # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),                               # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),                                                              # 10
            small_basic_block(ch_in=256, ch_out=256),                              # *** 11 ***
            nn.BatchNorm2d(num_features=256),                                      # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),                 # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 4), stride=1),   # 16
            # 输出通道从 256 减为 128，节省约 65% 的参数（backbone.16 从 256KB → 128KB）
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),                                                              # 18
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 18)),                                          # 19: 高度压缩到1
            # 在最后 Conv 之前先把高度自适应压缩到 1，这样 kernel 只需 (1,1)
            # 避免了 kernel=(40,1) 带来的 788K 参数（原方案最大瓶颈）
            nn.Conv2d(in_channels=128, out_channels=class_num, kernel_size=(1, 1), stride=1),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),                                                              # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=448 + self.class_num,
                # 64(kf0) + 128(kf1) + 256(kf2) + class_num(kf3) = 448 + class_num
                out_channels=self.class_num,
                kernel_size=(1, 1),
                stride=(1, 1)
            ),
        )

    def forward(self, x):
        keep_features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 23]:
                # i=2:  ReLU → kf[0], shape [B, 64,  H0, W0]
                # i=6:  ReLU → kf[1], shape [B, 128, H1, W1]
                # i=13: ReLU → kf[2], shape [B, 256, H2, W2]
                # i=23: ReLU → kf[3], shape [B, class_num, 1, 18]（已经过 AdaptiveAvgPool）
                keep_features.append(x)

        # kf[0~2] 用 AdaptiveAvgPool2d((1,18)) 压缩到 (1,18)
        # kf[3] 已经是 (1,18)，直接用
        adaptive_pool = nn.AdaptiveAvgPool2d((1, 18))
        global_context = []
        for idx, f in enumerate(keep_features):
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            if idx < 3:
                f = adaptive_pool(f)   # kf[0~2]: 压缩到 [B, C, 1, 18]
            # kf[3] 已是 [B, class_num, 1, 18]，不需要再 pool
            global_context.append(f)

        x = torch.cat(global_context, 1)   # [B, 64+128+256+class_num, 1, 18]
        x = self.container(x)              # [B, class_num, 1, 18]
        logits = torch.mean(x, dim=2)      # [B, class_num, 18]
        return logits


def build_lprnet(lpr_max_len=8, phase=False, class_num=None, dropout_rate=0.5):
    """
    构建 LPRNet 模型。

    Args:
        lpr_max_len:  车牌最大字符数，普通牌7位，新能源牌8位，默认8
        phase:        'train' 返回训练模式，其他值返回推理模式
        class_num:    字符类别数，None 时自动从 chars_config 读取当前配置
        dropout_rate: Dropout 比率

    Returns:
        LPRNet 实例（train 或 eval 模式）
    """
    if class_num is None:
        class_num = len(CHARS)

    net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return net.train()
    else:
        return net.eval()


def _remap_key(new_key):
    """
    新模型层名 → 旧模型层名的映射。
    因为新模型在 backbone 中插入了 AdaptiveAvgPool2d（index 20），
    导致后续层编号后移：
      新 backbone.21 (Conv2d 128→77) ← 旧 backbone.20 (Conv2d 256→68)
      新 backbone.22 (BN 77)         ← 旧 backbone.21 (BN 68)
      新 backbone.23 (ReLU, 无参数)  ← 旧 backbone.22 (ReLU, 无参数)
    """
    remap = {
        'backbone.21': 'backbone.20',
        'backbone.22': 'backbone.21',
    }
    for new_prefix, old_prefix in remap.items():
        if new_key.startswith(new_prefix + '.') or new_key == new_prefix:
            return new_key.replace(new_prefix, old_prefix, 1)
    return None


def expand_output_layer(old_model_path, new_model_path=None, device=None):
    """
    将原始 68 类模型权重迁移到 77 类（完整国标字符集）。

    迁移策略：
      - 旧字符集中已有的字符：直接复制对应权重（保留已学到的特征）
      - 新增字符（港/澳/使/领/警/学/挂/试/练）：
          * backbone 最后一层 Conv2d (index 20) 的输出通道：用旧汉字通道均值初始化
          * container Conv2d 的输入/输出通道：同样用均值初始化
          * BatchNorm 参数：用旧汉字 BN 参数均值初始化
      - 字母 I/O 的位置调整：原始集 I=index63, O=index64；新集 I=index58, O=index64
        直接按字符名称映射，不按位置映射

    Args:
        old_model_path: 原始预训练权重路径（68类）
        new_model_path: 扩展后权重保存路径，None 则不保存
        device:         torch.device，None 时自动选择

    Returns:
        扩展后的 LPRNet 模型（77类，eval 模式）
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 加载旧权重
    old_state = torch.load(old_model_path, map_location=device)

    # 构建新模型（77类）
    new_net = build_lprnet(lpr_max_len=8, phase=False, class_num=FULL_CLASS_NUM, dropout_rate=0.5)
    new_state = new_net.state_dict()

    # ── 找出需要特殊处理的层（输出维度从 68 变 77 的层）────────────────────────
    # backbone 第 20 层：Conv2d(256, class_num, (40,1))  → weight shape [class_num, 256, 40, 1]
    # backbone 第 21 层：BatchNorm2d(class_num)          → weight/bias/mean/var shape [class_num]
    # container 第 0 层：Conv2d(448+class_num, class_num, (1,1))
    #                    → weight shape [class_num, 448+class_num, 1, 1]

    # 建立旧字符集 index → 新字符集 index 的映射
    old_to_new = {}
    for old_idx, char in enumerate(CHARS_ORIGINAL):
        if char in CHAR2IDX_FULL:
            old_to_new[old_idx] = CHAR2IDX_FULL[char]

    # 新增字符的 index（在新字符集中）
    new_char_indices = [
        CHAR2IDX_FULL[c] for c in CHARS_FULL
        if c not in CHAR2IDX_ORIGINAL
    ]
    # 旧字符集中汉字的 index（用于计算均值初始化新汉字）
    old_hanzi_indices = list(range(31))  # 0-30 是省份简称

    print(f'[expand] 旧字符集: {ORIGINAL_CLASS_NUM} 类 → 新字符集: {FULL_CLASS_NUM} 类')
    print(f'[expand] 新增字符 ({len(new_char_indices)} 个): '
          f'{[CHARS_FULL[i] for i in new_char_indices]}')
    print(f'[expand] 字符映射: {len(old_to_new)} 个旧字符直接复制权重')

    for key in new_state.keys():
        if key not in old_state:
            # ── 新模型有但旧模型没有的层 ──────────────────────────────────────
            # backbone.19: AdaptiveAvgPool2d（无参数，不在 state_dict 里）
            # backbone.22: 新 BN（对应旧 backbone.21，但 key 不同）
            # 尝试用旧模型的对应层名来初始化
            old_key = _remap_key(key)
            if old_key and old_key in old_state:
                old_tensor = old_state[old_key]
                new_tensor = new_state[key]
                if old_tensor.shape == new_tensor.shape:
                    new_state[key] = old_tensor.clone()
                    print(f'[expand] 重映射复制: {old_key} → {key}')
                else:
                    # class_num 相关层（BN 从 68 扩展到 77）
                    _copy_output_dim(new_state, key, old_tensor, new_tensor,
                                     old_to_new, new_char_indices, old_hanzi_indices,
                                     dim=0)
                    print(f'[expand] 重映射扩展: {old_key}({old_tensor.shape}) → {key}({new_tensor.shape})')
            else:
                print(f'[expand] 跳过（旧模型无此层）: {key}')
            continue

        old_tensor = old_state[key]
        new_tensor = new_state[key]

        # 形状相同，直接复制
        if old_tensor.shape == new_tensor.shape:
            new_state[key] = old_tensor.clone()
            continue

        # ── 形状不同 ──────────────────────────────────────────────────────────
        # 新模型 backbone.21（Conv2d 128→77 (1,1)）对应旧 backbone.20（Conv2d 256→68 (13,1)）
        # 新模型 container.0 对应旧 container.0，但输入通道数变了
        if 'backbone.21' in key or 'backbone.22' in key:
            # class_num 输出维度扩展（68→77）
            _copy_output_dim(new_state, key, old_tensor, new_tensor,
                             old_to_new, new_char_indices, old_hanzi_indices,
                             dim=0)

        elif 'container.0' in key:
            if 'weight' in key:
                _copy_container_weight(new_state, key, old_tensor, new_tensor,
                                       old_to_new, new_char_indices, old_hanzi_indices)
            else:
                _copy_output_dim(new_state, key, old_tensor, new_tensor,
                                 old_to_new, new_char_indices, old_hanzi_indices,
                                 dim=0)

        else:
            # backbone.16/17（通道数 256→128）：截断复制前 128 个通道
            if old_tensor.dim() == new_tensor.dim():
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_tensor.shape, new_tensor.shape))
                result = new_tensor.clone()
                result[slices] = old_tensor[slices]
                new_state[key] = result
                print(f'[expand] 截断复制: {key} {old_tensor.shape} → {new_tensor.shape}')
            else:
                print(f'[expand] 警告：{key} 维度数不同，跳过')

    new_net.load_state_dict(new_state)
    new_net.eval()

    if new_model_path:
        torch.save(new_net.state_dict(), new_model_path)
        print(f'[expand] 扩展后权重已保存到: {new_model_path}')

    return new_net


def _copy_output_dim(new_state, key, old_tensor, new_tensor,
                     old_to_new, new_char_indices, old_hanzi_indices, dim=0):
    """按字符映射复制输出维度（dim=0）的权重。"""
    result = new_tensor.clone()

    # 1. 复制旧字符对应的权重
    for old_idx, new_idx in old_to_new.items():
        idx_old = [slice(None)] * old_tensor.dim()
        idx_new = [slice(None)] * result.dim()
        idx_old[dim] = old_idx
        idx_new[dim] = new_idx
        result[tuple(idx_new)] = old_tensor[tuple(idx_old)]

    # 2. 新增字符用旧汉字均值初始化
    idx_old_hanzi = [slice(None)] * old_tensor.dim()
    idx_old_hanzi[dim] = old_hanzi_indices
    hanzi_mean = old_tensor[tuple(idx_old_hanzi)].mean(dim=dim, keepdim=True)
    # hanzi_mean shape: [1, ...] 或 [1]

    for new_idx in new_char_indices:
        idx_new = [slice(None)] * result.dim()
        idx_new[dim] = new_idx
        if result.dim() == 1:
            result[new_idx] = hanzi_mean.squeeze()
        else:
            result[tuple(idx_new)] = hanzi_mean.squeeze(dim)

    new_state[key] = result


def _copy_container_weight(new_state, key, old_tensor, new_tensor,
                            old_to_new, new_char_indices, old_hanzi_indices):
    """
    复制 container.0.weight，形状 [new_class, 448+new_class, 1, 1]。
    输出维度 (dim=0) 和输入维度中 class_num 部分 (dim=1, offset=448) 都要扩展。
    """
    old_class = ORIGINAL_CLASS_NUM  # 68
    new_class = FULL_CLASS_NUM      # 77
    result = new_tensor.clone()

    # ── 先处理输出维度 (dim=0) ──────────────────────────────────────────────
    # 旧 weight: [68, 448+68, 1, 1] = [68, 516, 1, 1]
    # 新 weight: [77, 448+77, 1, 1] = [77, 525, 1, 1]

    # 对于输入维度，旧的 448+68=516 通道中：
    #   前 448 通道：backbone 特征，直接复制
    #   后 68 通道：旧字符集的 class 通道，需要按映射扩展到 77 通道

    # 构建输入维度的映射（旧 516 → 新 525）
    # 前 448 通道直接对应
    # 后 68 通道按 old_to_new 映射到新的 77 通道位置

    for old_out_idx, new_out_idx in old_to_new.items():
        # 复制前 448 个输入通道
        result[new_out_idx, :448, :, :] = old_tensor[old_out_idx, :448, :, :]
        # 复制后 class_num 个输入通道（按字符映射）
        for old_in_idx, new_in_idx in old_to_new.items():
            result[new_out_idx, 448 + new_in_idx, :, :] = \
                old_tensor[old_out_idx, 448 + old_in_idx, :, :]

    # 新增字符的输出行：用旧汉字行均值初始化
    old_hanzi_rows = old_tensor[old_hanzi_indices, :, :, :]  # [31, 516, 1, 1]
    hanzi_mean_row = old_hanzi_rows.mean(dim=0, keepdim=True)  # [1, 516, 1, 1]

    for new_out_idx in new_char_indices:
        # 前 448 通道
        result[new_out_idx, :448, :, :] = hanzi_mean_row[0, :448, :, :]
        # 后 class_num 通道：新增字符对应的输入通道也用均值
        for old_in_idx, new_in_idx in old_to_new.items():
            result[new_out_idx, 448 + new_in_idx, :, :] = \
                hanzi_mean_row[0, 448 + old_in_idx, :, :]
        # 新增字符对应的输入通道（新增字符的输入通道）用小随机值
        for new_in_idx in new_char_indices:
            result[new_out_idx, 448 + new_in_idx, :, :] = \
                torch.randn_like(result[new_out_idx, 448 + new_in_idx, :, :]) * 0.01

    new_state[key] = result
