// RUN: triton-opt %s --triton-to-linalg-incubated="named-ops=true" --canonicalize | FileCheck %s
#partition = #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>
tt.func public @view_passthrough(%base: !tv.ptr<f32>, %n: index, %stride: index, %i: index) {
  %v = tv.make_tensor_view %base, sizes = [%n], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
  %p = tv.make_partition_view %v : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #partition>
  %t = tv.view_load %p[%i] : !tv.tensor_view<?xf32, strides=[?], #partition>, index -> tensor<4xf32>
  %sum = arith.addf %t, %t : tensor<4xf32>
  tv.view_store %p[%i], %sum : !tv.tensor_view<?xf32, strides=[?], #partition>, tensor<4xf32>, index
  tt.return
}
// CHECK-LABEL: func.func @view_passthrough
// CHECK-SAME: !tv.ptr<f32>
// CHECK: tv.make_tensor_view
// CHECK: tv.make_partition_view
// CHECK: tv.view_load
// CHECK: linalg.
// CHECK: tv.view_store

tt.func public @ptr_passthrough(%base: !tt.ptr<f32>) {
  %offsets = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %bases = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %pointers = tt.addptr %bases, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %indices = arith.constant dense<[0, 1, 2, 3]> : tensor<4xindex>
  %t = tv.ptr_load %pointers, indices = [%indices] : tensor<4x!tt.ptr<f32>>, tensor<4xindex> -> tensor<4xf32>
  tv.ptr_store %pointers, %t, indices = [%indices] : tensor<4x!tt.ptr<f32>>, tensor<4xf32>, tensor<4xindex>
  tt.return
}
// CHECK-LABEL: func.func @ptr_passthrough
// CHECK: tv.ptr_load {{.*}} : tensor<4xi64>
// CHECK: tv.ptr_store {{.*}} : tensor<4xi64>
