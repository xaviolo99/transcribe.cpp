#!/usr/bin/env python3
"""GPU-less smoke of the step API wheels, run in a venv holding only them
and their dependencies: the CUDA provider is the one selected, it declares CUDA,
and every step API symbol resolves in the loaded library."""
import transcribe_cpp as t
from transcribe_cpp_native_cu12 import _contract

PROVIDER = "transcribe-cpp-native-cu12"
STEP_API = (
    "transcribe_whisper_dims_init",
    "transcribe_whisper_get_dims",
    "transcribe_whisper_step_set_align_heads",
    "transcribe_whisper_step_encode",
    "transcribe_whisper_step_decode",
    "transcribe_whisper_step_get_logits",
    "transcribe_whisper_step_get_cross_attention",
)

print("provider:", t.native_provider())
print("library: ", t._lib_path)
print("backends:", _contract.BACKENDS)
assert t.native_provider() == PROVIDER, t.native_provider()
assert "cuda" in _contract.BACKENDS, _contract.BACKENDS
for name in STEP_API:
    getattr(t._lib, name)
print("ok")
