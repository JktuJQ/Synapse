{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Vector-Jacobian products for reductions and normalization.
module Synapse.Autograd.Tensor.Ops.Pullback.Reduction
    ( -- * Sums and means
      sum
    , sumAxes
    , mean
    , meanAxes

      -- * Norms
    , normL1
    , normL2

      -- * Extrema
    , extremum
    , extremumAxes

      -- * Exponential reductions and normalization
    , softmax
    , logSoftmax
    , logSumExp
    , logSumExpAxes
    ) where

import Data.Proxy (Proxy (..))
import Prelude hiding (sum)

import Synapse.Autograd.Tensor.Ops.Pullback.Shape (axesOf, expandReduced, shapeOf, singletonAxis)
import Synapse.Tensor.Ops.Math ()
import Synapse.Tensor.Ops.Raw.Construction (element)
import Synapse.Tensor.Representation (DType, Tensor, TensorElement)
import Synapse.Tensor.Shapes.Axes (Axes, ReduceShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)

import qualified Synapse.Tensor.Ops.Elementwise as E
import qualified Synapse.Tensor.Ops.Raw.Elementwise as H
import qualified Synapse.Tensor.Ops.Raw.Reduction as Raw

-- | All axes removed by a full reduction.
allAxes
    :: Shape sh
    => Tensor sh a
    -> [Int]
allAxes input = [0 .. length (shapeOf input) - 1]

-- | Repeat a scalar output seed at every input position.
sum
    :: (Shape sh, DType a)
    => Tensor sh a
    -> Tensor '[] a
    -> Tensor '[] a
    -> Tensor sh a
sum input _ = expandReduced (allAxes input) input

-- | Repeat a seed over the selected reduction axes.
sumAxes
    :: forall xs sh a
     . (Axes xs, Shape sh, Shape (ReduceShape xs sh), DType a)
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor sh a
sumAxes input _ = expandReduced (axesOf (Proxy @xs)) input

-- | Divide each repeated output seed by the number of reduced elements.
mean
    :: forall sh a
     . (Shape sh, DType a, Num a, Fractional (TensorElement a))
    => Tensor sh a
    -> Tensor '[] a
    -> Tensor '[] a
    -> Tensor sh a
mean input output seed =
    let count = element (fromIntegral (shapeSize @sh) :: a)
     in H.map (/ count) (sum input output seed)

-- | Divide each repeated seed by its axis-wise reduction region size.
meanAxes
    :: forall xs sh a
     . (Axes xs, Shape sh, Shape (ReduceShape xs sh), DType a, Num a, Fractional (TensorElement a))
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor sh a
meanAxes input output seed =
    let count = element (fromIntegral (Raw.reductionSize @xs @sh) :: a)
     in H.map (/ count) (sumAxes @xs input output seed)

-- | Choose zero at the origin and otherwise multiply by the element's sign.
normL1
    :: (Shape sh, DType a, Num a, Num (TensorElement a))
    => Tensor sh a
    -> Tensor '[] a
    -> Tensor '[] a
    -> Tensor sh a
normL1 input _ seed = signum input * expandReduced (allAxes input) input seed

-- | Normalize the input direction, choosing a zero cotangent at the origin.
normL2
    :: (Shape sh, DType a, Fractional a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => Tensor sh a
    -> Tensor '[] a
    -> Tensor '[] a
    -> Tensor sh a
normL2 input output seed =
    let safeNorm = E.select (output E.== 0) 1 output
        selected = allAxes input
     in (input / expandReduced selected input safeNorm) * expandReduced selected input seed

-- | Divide an extremum's seed uniformly among all input positions tied at its value.
extremum
    :: (Shape sh, DType a, Fractional a, Fractional (TensorElement a), E.Ord (TensorElement a))
    => Tensor sh a
    -> Tensor '[] a
    -> Tensor '[] a
    -> Tensor sh a
extremum input output seed =
    let selected = allAxes input
        weights = H.zipWith (\x y -> E.select (x E.== y) 1 0) input (expandReduced selected input output)
        counts = Raw.sum weights
     in weights * expandReduced selected input (seed / counts)

-- | Share each extremum seed among ties within its selected reduction region.
extremumAxes
    :: forall xs sh a
     . ( Axes xs
       , Shape sh
       , Shape (ReduceShape xs sh)
       , DType a
       , Fractional a
       , Fractional (TensorElement a)
       , E.Ord (TensorElement a)
       )
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor sh a
extremumAxes input output seed =
    let selected = axesOf (Proxy @xs)
        weights = H.zipWith (\x y -> E.select (x E.== y) 1 0) input (expandReduced selected input output)
        counts = Raw.sumAxes @xs weights
     in weights * expandReduced selected input (seed / counts)

-- | Subtract the output-weighted seed sum along the normalization axis.
softmax
    :: forall axis sh a proxy
     . (Axes '[axis], Shape sh, Shape (ReduceShape '[axis] sh), DType a, Num a, Num (TensorElement a))
    => proxy axis
    -> Tensor sh a
    -> Tensor sh a
    -> Tensor sh a
    -> Tensor sh a
softmax selected input output seed =
    output * (seed - expandReduced (axesOf (singletonAxis selected)) input (Raw.sumAxes @'[axis] (seed * output)))

-- | Subtract the softmax-weighted seed sum along the normalization axis.
logSoftmax
    :: forall axis sh a proxy
     . ( Axes '[axis]
       , Shape sh
       , Shape (ReduceShape '[axis] sh)
       , DType a
       , Floating a
       , Floating (TensorElement a)
       )
    => proxy axis
    -> Tensor sh a
    -> Tensor sh a
    -> Tensor sh a
    -> Tensor sh a
logSoftmax selected input output seed =
    seed - exp output * expandReduced (axesOf (singletonAxis selected)) input (Raw.sumAxes @'[axis] seed)

-- | Weight a repeated scalar seed by the corresponding normalized exponential.
logSumExp
    :: (Shape sh, DType a, Floating a, Floating (TensorElement a))
    => Tensor sh a
    -> Tensor '[] a
    -> Tensor '[] a
    -> Tensor sh a
logSumExp input output seed =
    let selected = allAxes input
     in exp (input - expandReduced selected input output) * expandReduced selected input seed

-- | Weight repeated seeds by normalized exponentials within each reduction region.
logSumExpAxes
    :: forall xs sh a
     . (Axes xs, Shape sh, Shape (ReduceShape xs sh), DType a, Floating a, Floating (TensorElement a))
    => Tensor sh a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor (ReduceShape xs sh) a
    -> Tensor sh a
logSumExpAxes input output seed =
    let selected = axesOf (Proxy @xs)
     in exp (input - expandReduced selected input output) * expandReduced selected input seed
