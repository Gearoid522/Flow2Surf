"""CUDA source loading for Jittor custom operators.

jt.code source snippets place kernel declarations before host launch code;
device helpers and includes belong in their separate headers.
"""

from pathlib import Path

_CUDA_DIR = Path(__file__).parent / "cuda"


def load_cuda(name: str) -> str:
    return (_CUDA_DIR / name).read_text(encoding="utf-8")
