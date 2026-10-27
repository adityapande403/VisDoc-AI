import sys
import platform
import os
from pathlib import Path
import torch
import pymupdf
import transformers
from rich.console import Console
from rich.table import Table

console = Console()

def inspect():
    table = Table(title="VisDoc AI - Environment & Hardware Audit")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Detected Value", style="green")

    table.add_row("Python Executable", sys.executable)
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("OS Platform", f"{platform.system()} {platform.release()} ({platform.architecture()[0]})")
    table.add_row("PyTorch Version", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        table.add_row("CUDA Device Count", str(torch.cuda.device_count()))
        table.add_row("GPU Device Name", torch.cuda.get_device_name(0))
        cap = torch.cuda.get_device_capability(0)
        table.add_row("Compute Capability", f"sm_{cap[0]}{cap[1]}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        table.add_row("VRAM Total", f"{vram_gb:.2f} GB")
        table.add_row("BFloat16 Supported", str(torch.cuda.is_bf16_supported()))

    table.add_row("PyMuPDF Version", pymupdf.__version__)
    table.add_row("Transformers Version", transformers.__version__)
    table.add_row("HF_HOME Cache Dir", os.environ.get("HF_HOME", "Default (~/.cache)"))

    console.print(table)

if __name__ == "__main__":
    inspect()
