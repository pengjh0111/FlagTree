// RUN: triton-opt %s --tensor-view-to-hivm | FileCheck %s

// Gather keeps the GM base dynamically sized, but the indices and destination
// passed to HIVM have the compile-time physical tile shape.

// CHECK-LABEL: tt.func public @gather_last_dim
// CHECK-SAME: memref<?xf32, #hivm.address_space<gm>>
tt.func public @gather_last_dim(%src: !tv.ptr<f32>,
                                %cols: tensor<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %src, sizes = [%c4, %c12], strides = [%c12, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<4x12xf32, strides=[12, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<4x12xf32, strides=[12, 1]>
     -> !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>
  // CHECK-NOT: memref.alloc(%{{.*}}) : memref<{{.*}}?{{.*}}>
  // CHECK: %[[INIT_LAST:.*]] = tensor.empty() : tensor<4x4xf32>
  // CHECK: hivm.hir.gather_load
  // CHECK-SAME: tensor<4x4xi64>
  // CHECK-SAME: outs(%[[INIT_LAST]] : tensor<4x4xf32>)
  // CHECK-SAME: -> tensor<4x4xf32>
  %result = tv.view_load %view[%c0, %cols]
      : !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>, index, tensor<4xi32>
     -> tensor<4x4xf32>
  tt.return
}

// Non-last-dimension gather uses physical offsets and the same static-tile
// HIVM gather interface; its contiguous trailing dimension has burst length 4.

// CHECK-LABEL: tt.func public @gather_non_last_dim
tt.func public @gather_non_last_dim(%src: !tv.ptr<f32>,
                                    %rows: tensor<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %src, sizes = [%c12, %c4], strides = [%c4, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<12x4xf32, strides=[4, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<12x4xf32, strides=[4, 1]>
     -> !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.extract %{{.*}}[%{{.*}}] : tensor<4xi32>
  // CHECK: tensor.insert
  // CHECK: arith.constant 4 : i64
  // CHECK-NOT: memref.alloc(%{{.*}}) : memref<{{.*}}?{{.*}}>
  // CHECK: %[[INIT_NON_LAST:.*]] = tensor.empty() : tensor<4x4xf32>
  // CHECK: hivm.hir.gather_load
  // CHECK-SAME: tensor<4x4xi64>
  // CHECK-SAME: outs(%[[INIT_NON_LAST]] : tensor<4x4xf32>)
  %result = tv.view_load %view[%rows, %c0]
      : !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4xi32>, index
     -> tensor<4x4xf32>
  tt.return
}

// Scatter constructs a tile-shaped physical-offset tensor and emits the
// native discrete DMA store.  With sparse_dim=0, each row has a contiguous
// four-element burst.

// CHECK-LABEL: tt.func public @scatter_non_last_dim
tt.func public @scatter_non_last_dim(%dst: !tv.ptr<f32>,
                                     %rows: tensor<4xi32>,
                                     %value: tensor<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %dst, sizes = [%c12, %c4], strides = [%c4, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<12x4xf32, strides=[4, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<12x4xf32, strides=[4, 1]>
     -> !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.extract %{{.*}}[%{{.*}}] : tensor<4xi32>
  // CHECK: tensor.insert
  // CHECK: arith.constant 4 : i64
  // CHECK: hivm.hir.scatter_store
  // CHECK-SAME: tensor<4x4xi64>
  // CHECK-SAME: tensor<4x4xf32>
  tv.view_store %view[%rows, %c0], %value
      : !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4x4xf32>, tensor<4xi32>, index
  tt.return
}

// A last-dimension scatter has no contiguous elements after its sparse axis,
// therefore its conservative burst length is one.

// CHECK-LABEL: tt.func public @scatter_last_dim
tt.func public @scatter_last_dim(%dst: !tv.ptr<f32>,
                                 %cols: tensor<4xi32>,
                                 %value: tensor<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %dst, sizes = [%c4, %c12], strides = [%c12, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<4x12xf32, strides=[12, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<4x12xf32, strides=[12, 1]>
     -> !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>
  // CHECK: arith.constant 1 : i64
  // CHECK: hivm.hir.scatter_store
  tv.view_store %view[%c0, %cols], %value
      : !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>, tensor<4x4xf32>, index, tensor<4xi32>
  tt.return
}

// A masked gather sanitizes invalid offsets and delegates lane validity to the
// native masked gather.  Its destination remains a static physical tile.

// CHECK-LABEL: tt.func public @masked_gather_non_last_dim
// CHECK-NOT: arith.maxsi
// CHECK-NOT: hivm.hir.load
// CHECK: scf.if %{{.*}} -> (index)
// CHECK-NOT: memref.alloc(%{{.*}}) : memref<{{.*}}?{{.*}}>
// CHECK: %[[MASKED_INIT:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK: hivm.hir.gather_load
// CHECK-SAME: tensor<4x4xi64>
// CHECK-SAME: tensor<4x4xi1>
// CHECK-SAME: outs(%[[MASKED_INIT]] : tensor<4x4xf32>)
tt.func public @masked_gather_non_last_dim(
    %src: !tv.ptr<f32>, %rows: tensor<4xi32>, %mask: tensor<4x4xi1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %src, sizes = [%c12, %c4], strides = [%c4, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<12x4xf32, strides=[4, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<12x4xf32, strides=[4, 1]>
     -> !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>
  %result = tv.view_load %view[%rows, %c0], mask %mask : tensor<4x4xi1>
      : !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4xi32>, index
     -> tensor<4x4xf32>
  tt.return
}

// All-false is the important degenerate tail: even a deliberately invalid
// sparse index remains behind the condition and cannot cause a DMA/read.

// CHECK-LABEL: tt.func public @all_false_masked_gather_last_dim
// CHECK-NOT: arith.maxsi
// CHECK-NOT: hivm.hir.load
// CHECK-NOT: hivm.hir.vgather
// CHECK: scf.if %{{.*}} -> (index)
// CHECK-NOT: memref.alloc(%{{.*}}) : memref<{{.*}}?{{.*}}>
// CHECK: %[[FALSE_INIT:.*]] = tensor.empty() : tensor<4x4xf32>
// CHECK: hivm.hir.gather_load
// CHECK-SAME: tensor<4x4xi1>
// CHECK-SAME: outs(%[[FALSE_INIT]] : tensor<4x4xf32>)
tt.func public @all_false_masked_gather_last_dim(
    %src: !tv.ptr<f32>, %cols: tensor<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %false = arith.constant dense<false> : tensor<4x4xi1>
  %base = tv.make_tensor_view %src, sizes = [%c4, %c12], strides = [%c12, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<4x12xf32, strides=[12, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<4x12xf32, strides=[12, 1]>
     -> !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>
  %result = tv.view_load %view[%c0, %cols], mask %false : tensor<4x4xi1>
      : !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>, index, tensor<4xi32>
     -> tensor<4x4xf32>
  tt.return
}

// Scatter both sanitizes offsets for invalid lanes and passes the original
// mask to the hardware operation, so masked lanes never issue a store.

// CHECK-LABEL: tt.func public @masked_scatter_non_last_dim
// CHECK: scf.if %{{.*}} -> (index)
// CHECK: tensor.extract %{{.*}}[%{{.*}}] : tensor<4xi32>
// CHECK: scf.yield
// CHECK: scf.yield %{{.*}} : index
// CHECK: hivm.hir.scatter_store
// CHECK-SAME: tensor<4x4xi1>) outs
tt.func public @masked_scatter_non_last_dim(
    %dst: !tv.ptr<f32>, %rows: tensor<4xi32>, %value: tensor<4x4xf32>,
    %mask: tensor<4x4xi1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %base = tv.make_tensor_view %dst, sizes = [%c12, %c4], strides = [%c4, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<12x4xf32, strides=[4, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<12x4xf32, strides=[4, 1]>
     -> !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>
  tv.view_store %view[%rows, %c0], %value, mask %mask : tensor<4x4xi1>
      : !tv.tensor_view<12x4xf32, strides=[4, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [0], padding = zero>>, tensor<4x4xf32>, tensor<4xi32>, index
  tt.return
}

// The all-false scatter tail must keep the static offsets/data shape and pass
// the original lane mask to scatter_store.  Sanitized zero offsets are not
// observable because no masked-off lane may issue a write.

// CHECK-LABEL: tt.func public @all_false_masked_scatter_last_dim
// CHECK-NOT: hivm.hir.store
// CHECK: scf.if %{{[^ ]+}} -> (index)
// CHECK: arith.constant 1 : i64
// CHECK: hivm.hir.scatter_store
// CHECK-SAME: tensor<4x4xi64>
// CHECK-SAME: tensor<4x4xf32>
// CHECK-SAME: tensor<4x4xi1>) outs
tt.func public @all_false_masked_scatter_last_dim(
    %dst: !tv.ptr<f32>, %cols: tensor<4xi32>, %value: tensor<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %false = arith.constant dense<false> : tensor<4x4xi1>
  %base = tv.make_tensor_view %dst, sizes = [%c4, %c12], strides = [%c12, %c1]
      : !tv.ptr<f32> -> !tv.tensor_view<4x12xf32, strides=[12, 1]>
  %view = tv.make_gather_scatter_view %base
      : !tv.tensor_view<4x12xf32, strides=[12, 1]>
     -> !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>
  tv.view_store %view[%c0, %cols], %value, mask %false : tensor<4x4xi1>
      : !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>, tensor<4x4xf32>, index, tensor<4xi32>
  tt.return
}

// CHECK-NOT: tv.view_load
// CHECK-NOT: tv.view_store
// CHECK-NOT: tv.make_gather_scatter_view
// CHECK-NOT: tv.make_tensor_view
// CHECK-NOT: !tv.ptr
