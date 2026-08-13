// RUN: triton-opt %s --tensor-view-to-hivm | FileCheck %s

// Last-dimension gather: rectangular DMA to an untagged local buffer,
// rank expansion + broadcast of the sparse indices, then hardware vgather.

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
  // CHECK: memref.reinterpret_cast
  // CHECK-SAME: memref<?xf32, #hivm.address_space<gm>> to memref<4x?xf32, strided<[12, 1]
  // CHECK: memref.alloc(%{{.*}}) : memref<4x?xf32>
  // CHECK: hivm.hir.load
  // CHECK: tensor.expand_shape
  // CHECK-SAME: tensor<4xi32> into tensor<1x4xi32>
  // CHECK: hivm.hir.vbrc
  // CHECK-SAME: broadcast_dims = [0]
  // CHECK: hivm.hir.vgather
  // CHECK: bufferization.to_tensor
  %result = tv.view_load %view[%c0, %cols]
      : !tv.tensor_view<4x12xf32, strides=[12, 1], #tv.gather_scatter_view<tile = [4, 4], sparse_dim = [1], padding = zero>>, index, tensor<4xi32>
     -> tensor<4x4xf32>
  tt.return
}

// Non-last-dimension gather: the HIVM hardware op cannot express this axis,
// so use the same scalar loop decomposition as native AscendNPU-IR.

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
  // CHECK: hivm.hir.load
  // CHECK: bufferization.to_tensor
  // CHECK: scf.for
  // CHECK: scf.for
  // CHECK: tensor.extract %{{.*}}[%{{.*}}] : tensor<4xi32>
  // CHECK: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<?x4xf32>
  // CHECK: tensor.insert
  // CHECK-NOT: hivm.hir.vgather
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

// A masked gather must not derive a DMA span from invalid sparse indices.
// Instead, only the true branch performs the scalar GM access and false lanes
// retain zero padding.

// CHECK-LABEL: tt.func public @masked_gather_non_last_dim
// CHECK-NOT: arith.maxsi
// CHECK-NOT: hivm.hir.load
// CHECK: scf.if %{{.*}} -> (f32)
// CHECK: memref.reinterpret_cast
// CHECK: memref.load
// CHECK: scf.yield
// CHECK: scf.yield %{{.*}} : f32
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
// CHECK: scf.if %{{.*}} -> (f32)
// CHECK: memref.load
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

// CHECK-NOT: tv.view_load
// CHECK-NOT: tv.view_store
// CHECK-NOT: tv.make_gather_scatter_view
// CHECK-NOT: tv.make_tensor_view
// CHECK-NOT: !tv.ptr
