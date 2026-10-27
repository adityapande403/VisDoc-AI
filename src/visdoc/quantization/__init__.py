from visdoc.quantization.base import BaseQuantizer
from visdoc.quantization.fp16 import FP16Quantizer
from visdoc.quantization.int8 import INT8Quantizer
from visdoc.quantization.binary import BinaryQuantizer

def get_quantizer(format_name: str, **kwargs) -> BaseQuantizer:
    format_lower = format_name.lower()
    if format_lower == "fp16":
        return FP16Quantizer()
    elif format_lower == "int8":
        return INT8Quantizer(**kwargs)
    elif format_lower == "binary":
        return BinaryQuantizer(**kwargs)
    else:
        raise ValueError(f"Unknown quantization format: {format_name}. Choose 'fp16', 'int8', or 'binary'.")

__all__ = [
    "BaseQuantizer",
    "FP16Quantizer",
    "INT8Quantizer",
    "BinaryQuantizer",
    "get_quantizer",
]
