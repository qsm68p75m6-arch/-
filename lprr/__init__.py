from .LPRNet import LPRNet, build_lprnet, expand_output_layer, INPUT_W, INPUT_H
from .chars_config import (
    CHARS,                  # 始终是 CHARS_FULL（77类）
    CHARS_ORIGINAL,
    CHARS_FULL,
    PROVINCE_CHARS,
    SPECIAL_CHARS,
    ALL_FIRST_CHARS,
    ORIGINAL_CLASS_NUM,
    FULL_CLASS_NUM,
)
from .plate import de_lpr, decode_plate_number, reset_lprnet
