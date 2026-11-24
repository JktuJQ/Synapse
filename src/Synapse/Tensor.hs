{- | Statically shaped tensors and differentiable tensor operations.

This module is intended to be imported qualified:

@
import qualified Synapse.Tensor as T
@

= Type-level tensor shapes

A tensor's shape is part of its type:
@Tensor '[2, 3] Float@ describes a two-dimensional tensor with two rows and three columns,
while @Tensor '[] Float@ describes a scalar.
Shape constraints check the compatibility of
operations such as 'matmul', 'reshape', and 'broadcast' at compile time.

= Staged computation

Tensor computations are staged: applying an operation constructs a description of a computation
rather than immediately calculating its elements.
A 'Tensor' therefore represents a computation graph.
Haskell functions compose these graphs:
'transpose', 'matmul', and 'sum' can describe a complete computation
before any of its results are materialized.

'TensorData' holds materialized tensor values.
It is the boundary between host data and staged computation:
'fromListData' prepares input data,
'fromData' embeds data in a graph,
and 'toList' reads materialized results.
Calling 'run' on a closed graph executes it on the selected 'Backend' and returns its 'TensorData'.
The backend compiles and optimizes the graph, allowing operations to be fused
rather than requiring an intermediate array for every expression.

With a staged function, 'run' prepares it for repeated execution and preserves its argument list.
Bind the returned function once and pass changing data as its arguments.
Captured values remain fixed; preparation may compile lazily on the first invocation.
Shapes remain part of the types of both the computation and its materialized inputs and outputs.

= Working with tensors

An affine transformation @xW + b@ combines matrix multiplication
with an explicit broadcast of the bias across the batch.
Here each input has three features and each output has two:

@
affine
    :: T.Tensor '[3, 2] Float
    -> T.Tensor '[2] Float
    -> T.Tensor '[4, 3] Float
    -> T.Tensor '[4, 2] Float
affine weights bias input =
    (input `T.matmul` weights) + T.broadcast bias

-- Prepare once; weights, bias, and inputs can change between invocations.
step = T.run T.defaultCPU affine

weights :: T.TensorData '[3, 2] Float
weights = T.fromListData $ concat
    [ [1, 0]
    , [0, 1]
    , [1, 1]
    ]

bias :: T.TensorData '[2] Float
bias = T.fromListData [0.5, -0.5]

input :: T.TensorData '[4, 3] Float
input = T.fromListData $ concat
    [ [1, 2, 3]
    , [4, 5, 6]
    , [7, 8, 9]
    , [10, 11, 12]
    ]

-- [4.5, 4.5, 10.5, 10.5, 16.5, 16.5, 22.5, 22.5]
output = T.toList $ step weights bias input
@

Reductions can be composed with the same transformation inside the staged function.
For example, reduce the output features to one total per input:

@
featureTotals
    :: T.Tensor '[3, 2] Float
    -> T.Tensor '[2] Float
    -> T.Tensor '[4, 3] Float
    -> T.Tensor '[4] Float
featureTotals weights bias input =
    T.sumAxes @'[1] $ affine weights bias input
@

These examples use the @DataKinds@ and @TypeApplications@ language extensions.

= Tensor polymorphism and automatic differentiation

The tensor operations are polymorphic in the tensor representation through 'TensorOps'.
A function can use ordinary 'Tensor' values for evaluation
or the differentiable tensor wrapper from "Synapse.Autograd.Tensor" for automatic differentiation.
The wrapper records the operations needed to propagate gradients,
so the computation itself can be shared between both uses:

@
affineWithGrad
    :: (T.TensorOps t Float, Num (t '[4, 2] Float))
    => t '[3, 2] Float
    -> t '[2] Float
    -> t '[4, 3] Float
    -> t '[4, 2] Float
affineWithGrad weights bias input =
    (input `T.matmul` weights) + T.broadcast bias
@
-}
module Synapse.Tensor
    ( -- * Tensor representations
      module Synapse.Tensor.Representation
    , TensorStructure (WithTensor)

      -- * Shapes
    , module Synapse.Tensor.Shapes

      -- * Operations
    , module Synapse.Tensor.Ops

      -- * Data conversion
    , module Synapse.Tensor.Data

      -- * Backend execution
    , module Synapse.Tensor.Backend
    ) where

import Synapse.Tensor.Backend
import Synapse.Tensor.Data
import Synapse.Tensor.Ops
import Synapse.Tensor.Representation
    ( DType
    , Tensor
    , TensorData
    , TensorElement
    , TensorIndex
    , TensorLike
    )
import Synapse.Tensor.Shapes
import Synapse.Tensor.Structure (TensorStructure (WithTensor))
