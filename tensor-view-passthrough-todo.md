# TensorView passthrough implementation

The template functions in this change are compilation stubs. They do not
implement TensorView memory access and must not be used for numerical validation.

- [x] Verify the FlagTree/flir branch revisions and the AscendNPU-IR dependency.
- [x] Prepare `AscendNPU-IR-flagtree` on a `tensorview` branch at that dependency.
- [x] Add `enable_tensor_view_decompose` (default true) and preserve TensorView
      through FlagTree's adapter conversion when false.
- [x] Make the TensorView IR library reusable by AscendNPU-IR and register it in
      the relevant compiler tools without pulling in the Triton frontend.
- [x] Implement TensorView bufferization using the existing one-shot pipeline,
      preserving view construction and encoding information.
- [x] Add a TensorView-to-template pass next to HIVM-to-standard conversion;
      lower view chains and the pointer ABI before handing IR to HIVMC.
- [x] Add template stubs and integrate their exported symbols into the existing
      template bitcode build.
- [x] Add focused IR/pipeline tests, run the checks supported by this environment,
      and record exact verification results and remaining limitations.

## Revisions

- FlagTree: `941ee0f822cfdec2f39868cae597f583dfaaea7c` (`flir_tensor_view`).
- FLIR: `e467363055526618c46914a3f9648d159edeecb5`
  (`fix/tensor-view-static-strides`).
- AscendNPU-IR: `a205c9574907907d608da6029403415ca2f98d3c`, from
  `flagos-ai/FlagTree-AscendNPU-IR`, on the local `tensorview` branch in
  `../AscendNPU-IR-flagtree`. This is FlagTree's default CANN 9.0 dependency.

## Local build inputs

Pass these CMake cache entries when building FlagTree so the existing nested
checkouts are not used:

```text
-DASCENDNPU_IR_SRC_DIR=/Users/pengjunhui/Documents/workspace/AscendNPU-IR-flagtree
-DFLIR_SOURCE_DIR=/Users/pengjunhui/Documents/workspace/flir
```

When building AscendNPU-IR separately, pass
`-DBISHENGIR_TENSOR_VIEW_SOURCE_DIR=/Users/pengjunhui/Documents/workspace/flir`.
Build/install the template library with the normal `BISHENGIR_BUILD_TEMPLATE=ON`
and `BISHENG_COMPILER_PATH` settings, then use that installation's tools and
bitcode libraries together. The new pass is enabled by
`--enable-tensor-view-decompose=false`; Python kernels use
`enable_tensor_view_decompose=False`. The default remains the existing FLIR
lowering path. SIMT-only compilation is not supported by the passthrough path.

The pure dialect and frontend builders share source definitions. The backend
registers its bufferization/core models only in a full compiler build; FlagTree's
IR-only embedding does not depend on backend transformation libraries.

## Stub boundary

- View constructors and their encodings remain until template conversion.
  `tv.ptr` kernel arguments retain the dynamic-pointer ABI marking used by the
  backend, and are converted to GM memrefs during the final template pass.
- One-shot bufferization creates `tv.view_load_into`/`tv.view_store_from` and
  their pointer-access counterparts. Buffer operations use destination-style
  interfaces so existing memory planning can track their buffers.
- Template conversion supports directly constructed views of rank 1 through 4;
  sparse coordinates remain explicit buffers. Passing an encoded view through
  a function or control-flow result is not implemented.
- Template entry points use scalar i64 descriptors: base/data addresses,
  element-type code, padding, sparse-dimension mask, and per-dimension extents,
  element strides, tiles, traversal strides, local extents/strides, coordinates,
  sparse-index addresses/strides/bit widths. Pointer accesses pass zipped
  coordinate descriptors. This preserves the information needed by future
  template implementations without allocating descriptor arrays after memory
  planning. The scalar-only declarations call the exported symbol directly;
  they do not request an additional `llvm.emit_c_interface` wrapper. Scalar
  call lowering to LLVM was checked to retain the exact exported symbol.
- The exported `_mlir_ciface_tensor_view_*` functions are empty. They do not
  initialize load buffers or write stores. DMA pipe/synchronization support for
  real templates is intentionally not supplied by these compilation stubs.
  They must be implemented before running kernels for numerical validation.
- The automatic dependency pins must be updated to the user's published FLIR
  and AscendNPU-IR commits after the implementation is pushed. The local CMake
  source overrides above select the uncommitted implementation now.

## Verification

Completed on 2026-10-20:

- Built the shared TensorView dialect with LLVM 19.1.7 and LLVM 22.1.8.
- Built the actual AscendNPU-IR HIVM dialect and the new bufferization/template
  conversion sources in an isolated LLVM 19 test driver. The stock LLVM test
  installation uses a temporary alias for `applyPatternsGreedily`; the backend's
  existing LLVM patch 0055 supplies that API in a normal build.
- Ran the bufferization, partition/strided/gather-scatter template-conversion,
  pointer-access and unbufferized-input diagnostic checks in
  `bishengir/test/Conversion/TensorViewToTemplate` through that driver.
- Verified kernel pointer ABI marking and the program-id conversion → one-shot
  bufferization → template-call sequence, including the existing kernel-argument
  pass regression test.
- Checked all 16 view/ptr load/store entry points for ranks 1 through 4 against
  the signatures exported by host-compiled template bitcode.
- Passed the shared dialect's buffer-op parsing, verifier and padding checks.
- Passed LLVM 22 to LLVM 19 bytecode round trips with version 0, including
  function symbols, view encodings and negative-infinity padding. The default
  bytecode encoding was observed to corrupt a function symbol across these
  stock versions, so passthrough explicitly requests version 0.
- Generated the real backend option tables and verified the CLI option and
  `HIVMPipelineOptions.enableTensorViewDecompose` field.
- Ran all 9 Python pipeline-option tests with compiler runtime imports isolated
  and MLIR/tool execution mocked. This checks the actual changed Python
  functions, not NPU execution.
- Compiled the frontend TensorViewBuilder object and the changed tensor-kind
  marking object. Checked the pointer-address normalization helper's C++ syntax.

Not verified here:

- The full FlagTree package build requires its supported LLVM distribution.
  The stock LLVM 22 build encounters existing `GEN_PASS_CLASSES` and dataflow
  API incompatibilities; macOS also encounters the existing f2reduce `version`
  filename collision. No unrelated source changes were made to bypass these.
- The new frontend adapter lit test is provided but was not run with the vendor
  `triton-opt`. In particular, the complete JIT-to-adapter path remains to be
  checked with that toolchain.
- The complete optimization pipeline and HIVMC binary link were not executed.
  CANN/ccec/HIVMC and NPU hardware are unavailable. Template bitcode was checked
  using host Clang with the device annotation and device utility header disabled,
  not the CANN compiler.

The stub path is therefore an implementation and IR-level integration result,
not a claim of a validated executable NPU kernel. Its first downstream check
should compile the adapter lit test and a small kernel with the intended vendor
LLVM/CANN tools, then inspect the IR immediately before HIVMC.
