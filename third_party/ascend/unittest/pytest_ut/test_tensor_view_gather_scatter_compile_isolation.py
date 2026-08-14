# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Minimal hardware compile isolation for TensorView gather/scatter.

Each case intentionally contains at most one discrete HIVM memory operation:
  dense_baseline: no gather_load/scatter_store
  gather_only:    one gather_load, followed by a regular dense store
  scatter_only:   one regular dense load, followed by one scatter_store
"""

import os

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


ROWS = 12
COLS = 4
TILE = 4
USE_BYTECODE = os.environ.get("TV_GS_DIAG_USE_BYTECODE", "1") != "0"


@triton.jit
def tensor_view_dense_compile_baseline(src_ptr, dst_ptr,
                                       COLS: tl.constexpr,
                                       TILE: tl.constexpr):
    lanes = tl.arange(0, TILE)
    offsets = lanes[:, None] * COLS + lanes[None, :]
    values = tl.load(src_ptr + offsets)
    tl.store(dst_ptr + offsets, values)


@triton.jit
def tensor_view_gather_only_compile_isolation(src_ptr, index_ptr, dst_ptr,
                                              COLS: tl.constexpr,
                                              TILE: tl.constexpr):
    lanes = tl.arange(0, TILE)
    sparse_rows = tl.load(index_ptr + lanes)
    sparse_offsets = sparse_rows[:, None] * COLS + lanes[None, :]
    values = tl.load(src_ptr + sparse_offsets)

    dense_offsets = lanes[:, None] * COLS + lanes[None, :]
    tl.store(dst_ptr + dense_offsets, values)


@triton.jit
def tensor_view_scatter_only_compile_isolation(src_ptr, index_ptr, dst_ptr,
                                               COLS: tl.constexpr,
                                               TILE: tl.constexpr):
    lanes = tl.arange(0, TILE)
    dense_offsets = lanes[:, None] * COLS + lanes[None, :]
    values = tl.load(src_ptr + dense_offsets)

    sparse_rows = tl.load(index_ptr + lanes)
    sparse_offsets = sparse_rows[:, None] * COLS + lanes[None, :]
    tl.store(dst_ptr + sparse_offsets, values)


@pytest.mark.parametrize(
    "case", ["dense_baseline", "gather_only", "scatter_only"])
def test_tensor_view_gather_scatter_compile_isolation(case):
    device = torch.device("npu")
    src_cpu = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
    indices_cpu = torch.tensor([9, 2, 7, 0], dtype=torch.int32)
    src = src_cpu.to(device)
    indices = indices_cpu.to(device)

    if case == "dense_baseline":
        dst = torch.full((TILE, COLS), -1.0, dtype=src.dtype, device=device)
        tensor_view_dense_compile_baseline[(1, )](
            src, dst, COLS=COLS, TILE=TILE, use_tensor_view=True,
            use_bytecode=USE_BYTECODE)
        expected = src_cpu[:TILE, :]
    elif case == "gather_only":
        dst = torch.full((TILE, COLS), -1.0, dtype=src.dtype, device=device)
        tensor_view_gather_only_compile_isolation[(1, )](
            src, indices, dst, COLS=COLS, TILE=TILE, use_tensor_view=True,
            use_bytecode=USE_BYTECODE)
        expected = torch.index_select(src_cpu, 0, indices_cpu.to(torch.int64))
    else:
        dst = torch.full_like(src, -1.0)
        tensor_view_scatter_only_compile_isolation[(1, )](
            src, indices, dst, COLS=COLS, TILE=TILE, use_tensor_view=True,
            use_bytecode=USE_BYTECODE)
        expected = torch.full_like(src_cpu, -1.0)
        expected[indices_cpu.to(torch.int64), :] = src_cpu[:TILE, :]

    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), expected, rtol=0.0, atol=0.0)
