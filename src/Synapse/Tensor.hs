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
Calling 'run' executes a graph on the selected 'Backend' and returns 'TensorData'.
The backend compiles and optimizes the graph, allowing operations to be fused
rather than requiring an intermediate array for every expression.

'run1' through 'run9' prepare staged functions for repeated execution.
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
step = T.run3 T.CPU affine

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
    ( -- * Tensor types
      Tensor
    , TensorData
    , TensorElement
    , TensorLike
    , TensorOps
    , BackwardConstraints
    , WithTensor
    , DType
    , Float32
    , Float64
    , Element

      -- * Shapes
    , Shape
    , shape
    , shapeSize
    , shapeRank

      -- * Construction
    , arange
    , eye
    , fill
    , fromGen
    , full
    , fullLike
    , identity
    , linspace
    , logspace
    , ones
    , onesLike
    , scalar
    , zeros
    , zerosLike

      -- * Conversion
    , FromIntegral
    , element
    , fromIntegral
    , fromListData
    , fromList
    , unsafeFromList
    , fromData
    , toList

      -- * Backend execution
    , Backend (..)
    , run
    , run1
    , run2
    , run3
    , run4
    , run5
    , run6
    , run7
    , run8
    , run9

      -- * Elementwise operations
    , Eq
    , Ord
    , Boolean
    , Integral
    , RealFloat
    , Conditional
    , quot
    , rem
    , isNaN
    , isInfinite
    , not
    , (&&)
    , (||)
    , select
    , cond
    , (==)
    , (/=)
    , (<)
    , (<=)
    , (>)
    , (>=)
    , min
    , max

      -- * Shape transformations
    , reshape
    , broadcast
    , squeeze
    , unsqueeze
    , transpose
    , permuteAxes
    , concat
    , stack

      -- * Indexing
    , TensorIndex
    , Indexable
    , SliceAxis (..)
    , coordinate
    , index
    , slice
    , (!!)
    , unsafeDynamicIndex
    , dynamicIndexOr
    , unsafeGather
    , gatherOr
    , unsafeLookupRows

      -- * Reductions
    , all
    , allAxes
    , any
    , anyAxes
    , sum
    , sumAxes
    , mean
    , meanAxes
    , normL1
    , normL2
    , minimum
    , minimumAxes
    , maximum
    , maximumAxes
    , argmax
    , argmin
    , softmax
    , logSoftmax
    , logSumExp
    , logSumExpAxes

      -- * Linear algebra
    , dot
    , matmul

      -- * Convolution
    , conv1d
    , conv2d
    , conv3d
    , convTranspose1d
    , convTranspose2d
    , convTranspose3d

      -- * Pooling
    , maxPool1d
    , maxPool2d
    , maxPool3d
    , avgPool1d
    , avgPool2d
    , avgPool3d

      -- * Patches
    , extractPatches1d
    , extractPatches2d
    , extractPatches3d
    , combinePatches1d
    , combinePatches2d
    , combinePatches3d

      -- * Random generation
    , StagedGenerator
    , generatorFromSeed
    , initializeGenerator
    , randomWords
    , uniform
    , bernoulli
    )
where

import Synapse.Tensor.Backend
    ( Backend (..)
    , run
    , run1
    , run2
    , run3
    , run4
    , run5
    , run6
    , run7
    , run8
    , run9
    )
import Synapse.Tensor.Data
    ( fromData
    , fromList
    , fromListData
    , toList
    , unsafeFromList
    )
import Synapse.Tensor.Ops.Construction
    ( arange
    , element
    , eye
    , fill
    , fromGen
    , full
    , fullLike
    , identity
    , linspace
    , logspace
    , ones
    , onesLike
    , scalar
    , zeros
    , zerosLike
    )
import Synapse.Tensor.Ops.Definition
    ( TensorOps (BackwardConstraints)
    )
import Synapse.Tensor.Ops.Elementwise
    ( Boolean
    , Conditional
    , Eq
    , FromIntegral
    , Integral
    , Ord
    , RealFloat
    , cond
    , fromIntegral
    , isInfinite
    , isNaN
    , max
    , min
    , not
    , quot
    , rem
    , select
    , (&&)
    , (/=)
    , (<)
    , (<=)
    , (==)
    , (>)
    , (>=)
    , (||)
    )
import Synapse.Tensor.Ops.Index
    ( Indexable
    , coordinate
    , dynamicIndexOr
    , gatherOr
    , index
    , slice
    , unsafeDynamicIndex
    , unsafeGather
    , unsafeLookupRows
    , (!!)
    )
import Synapse.Tensor.Ops.Math
    ( dot
    , matmul
    )
import Synapse.Tensor.Ops.Random
    ( StagedGenerator
    , bernoulli
    , generatorFromSeed
    , initializeGenerator
    , randomWords
    , uniform
    )
import Synapse.Tensor.Ops.Reduction
    ( all
    , allAxes
    , any
    , anyAxes
    , argmax
    , argmin
    , logSoftmax
    , logSumExp
    , logSumExpAxes
    , maximum
    , maximumAxes
    , mean
    , meanAxes
    , minimum
    , minimumAxes
    , normL1
    , normL2
    , softmax
    , sum
    , sumAxes
    )
import Synapse.Tensor.Ops.Shape
    ( broadcast
    , concat
    , permuteAxes
    , reshape
    , squeeze
    , stack
    , transpose
    , unsqueeze
    )
import Synapse.Tensor.Ops.Window
    ( avgPool1d
    , avgPool2d
    , avgPool3d
    , combinePatches1d
    , combinePatches2d
    , combinePatches3d
    , conv1d
    , conv2d
    , conv3d
    , convTranspose1d
    , convTranspose2d
    , convTranspose3d
    , extractPatches1d
    , extractPatches2d
    , extractPatches3d
    , maxPool1d
    , maxPool2d
    , maxPool3d
    )
import Synapse.Tensor.Representation
    ( DType
    , Element
    , Float32
    , Float64
    , Tensor
    , TensorData
    , TensorElement
    , TensorIndex
    , TensorLike
    , WithTensor
    )
import Synapse.Tensor.Shapes
    ( Shape
    , shape
    , shapeRank
    , shapeSize
    )
import Synapse.Tensor.Shapes.Index
    ( SliceAxis (..)
    )
import Prelude hiding
    ( Eq
    , Integral
    , Ord
    , RealFloat
    , all
    , any
    , concat
    , fromIntegral
    , isInfinite
    , isNaN
    , max
    , maximum
    , min
    , minimum
    , not
    , quot
    , rem
    , sum
    , (!!)
    , (&&)
    , (/=)
    , (<)
    , (<=)
    , (==)
    , (>)
    , (>=)
    , (||)
    )
