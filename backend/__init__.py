"""DiffusionCompress backend package.

On Windows, user-scoped CPU PyTorch wheels place MKL DLLs under the user
``Library/bin`` directory, which Python does not add to its DLL search path.
Register it before any backend module imports torch. This is harmless on other
platforms and keeps the app's documented installation reproducible.
"""
import os
import site
from pathlib import Path

# Diffusers uses PyTorch here. Prevent an unrelated, broken TensorFlow install
# from being imported by Transformers' optional image-processing helpers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

if os.name == "nt":
    _dll_handles = []
    _dll_handles.append(os.add_dll_directory(str(Path(__import__('sys').prefix))))
    # Some Windows Python installs lack the UCRT namespace DLLs in System32;
    # Visual Studio Build Tools provides the same runtime dependencies used by
    # the official CPU PyTorch wheel.
    visual_cpp = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\x64\bin")
    if visual_cpp.is_dir():
        _dll_handles.append(os.add_dll_directory(str(visual_cpp)))
    for user_site in site.getusersitepackages() if isinstance(site.getusersitepackages(), list) else [site.getusersitepackages()]:
        candidate = Path(user_site).parents[1] / "Library" / "bin"
        if candidate.is_dir():
            _dll_handles.append(os.add_dll_directory(str(candidate)))
        torch_lib = Path(user_site) / "torch" / "lib"
        if torch_lib.is_dir():
            _dll_handles.append(os.add_dll_directory(str(torch_lib)))
