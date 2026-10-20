"""TensorView pipeline selection; no NPU execution is required."""

from unittest.mock import MagicMock

import pytest

from triton.backends.ascend import compiler


def test_default_decomposes_tensor_view():
    assert compiler.NPUOptions().enable_tensor_view_decompose


def test_passthrough_has_a_separate_cache_key(monkeypatch):
    monkeypatch.setattr(compiler, "get_cann_version_file_hash", lambda: "test")
    assert compiler.NPUOptions().hash() != compiler.NPUOptions(enable_tensor_view_decompose=False).hash()


def test_passthrough_rejects_simt_only():
    with pytest.raises(ValueError, match="SIMD compilation pipeline"):
        compiler.NPUOptions(enable_tensor_view_decompose=False, compile_mode="simt_only")


@pytest.mark.parametrize("decompose", [True, False])
def test_adapter_selects_tensor_view_lowering(monkeypatch, decompose):
    ascend = MagicMock()
    monkeypatch.setattr(compiler, "ascend", ascend)
    monkeypatch.setattr(compiler, "passes", MagicMock())
    monkeypatch.setattr(compiler, "ir", MagicMock())
    monkeypatch.setattr(compiler, "_flagprism", MagicMock())
    monkeypatch.setattr(compiler, "_get_triton_adapter_opt_path", lambda: "/unused")
    monkeypatch.setattr(compiler, "_is_auto_map_parallel_blocks_enabled", lambda: False)
    metadata = dict.fromkeys([
        "enable_nd2nz_on_vector", "enable_select_analysis", "compile_on_910_95",
        "force_simt_template", "enable_sync_block_lock", "enable_mask_fallback_conversion",
        "optimize_dynamic_offset", "add_auto_scheduling", "enable_dynamic_cv_pipeline",
    ], False)
    metadata["auto_blockify_size"] = 1
    options = compiler.NPUOptions(enable_tensor_view_decompose=decompose)
    compiler.ttir_to_linalg(MagicMock(), metadata, options)
    assert ascend.passes.ttir.add_tensor_view_lowering.call_count == int(decompose)


@pytest.mark.parametrize("decompose", [True, False])
def test_backend_receives_tensor_view_option(decompose):
    options = []
    compiler._append_custom_pipeline_option(options, {"enable_tensor_view_decompose": decompose})
    assert ("--enable-tensor-view-decompose=false" in options) == (not decompose)


@pytest.mark.parametrize("decompose", [True, False])
def test_bytecode_uses_compatible_operation_properties(monkeypatch, decompose):
    monkeypatch.setattr(compiler, "_get_triton_mlir_opt_path", lambda: "/unused")

    def emit_bytecode(command, **kwargs):
        from pathlib import Path
        assert ("--emit-bytecode-version=0" in command) == (not decompose)
        Path(command[-1]).write_bytes(b"bytecode")

    monkeypatch.setattr(compiler.subprocess, "run", emit_bytecode)
    options = compiler.NPUOptions(enable_tensor_view_decompose=decompose)
    assert compiler.linalg_to_bc_by_triton_mlir_opt("module {}", {}, options) == b"bytecode"
