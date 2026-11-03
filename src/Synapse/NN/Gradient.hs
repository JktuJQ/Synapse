{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Global gradient norms and clipping, independent of optimizer updates.
-- Every tensor field contributes to one norm; clipping applies the same factor
-- to all fields. Operations build tensor expressions without host execution.
module Synapse.NN.Gradient
  ( gradientNormL1,
    gradientNormL2,
    clipGradNormL1,
    clipGradNormL2,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Bifunctor (bimap)
import Synapse.Autograd.Execution.Variable (ValueTree (..), Values (..))
import Synapse.Autograd.Type (Differentiable, ParametersConstraints)
import qualified Synapse.Tensor as T
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Type (Tensor (..))

-- | Tensor fields supported by global gradient operations.
class GradientTensor tensor where
  reduceGradient :: (forall sh. (T.Shape sh) => Tensor sh Double -> Tensor '[] Double) -> tensor -> Tensor '[] Double
  scaleGradient :: Tensor '[] Double -> tensor -> tensor

instance (T.Shape sh) => GradientTensor (Tensor sh Double) where
  reduceGradient reduce = reduce
  scaleGradient factor gradient = T.broadcast factor * gradient

instance (T.Shape sh) => GradientTensor (Tensor sh Float) where
  reduceGradient reduce (Tensor input) = reduce (Tensor (A.map A.toFloating input) :: Tensor sh Double)
  scaleGradient (Tensor factor) (Tensor input) =
    Tensor $ A.map (\value -> A.toFloating ((A.toFloating value :: A.Exp Double) * A.the factor)) input

-- | Fold tensor fields without requiring numeric instances for their records.
foldGradients ::
  forall p.
  (Differentiable p, ParametersConstraints GradientTensor p) =>
  (forall tensor. (GradientTensor tensor) => tensor -> Tensor '[] Double) ->
  (Tensor '[] Double -> Tensor '[] Double -> Tensor '[] Double) ->
  p ->
  Tensor '[] Double
foldGradients reduce combine =
  let visit :: ValueTree GradientTensor fields -> fields -> Tensor '[] Double
      visit TensorValues = reduce
      visit (PairValues left right) = \(x, y) -> combine (visit left x) (visit right y)
   in visit (values @(Leaves Tensor p) @GradientTensor) . flatten @Tensor

-- | Apply one shared clipping factor while preserving all tensor shapes and dtypes.
clipGradNorm ::
  forall p.
  (Differentiable p, ParametersConstraints GradientTensor p) =>
  (p -> Tensor '[] Double) ->
  Double ->
  p ->
  p
clipGradNorm norm maxNorm gradients =
  let magnitude = norm gradients
      limit = T.full maxNorm
      safeNorm = T.select (magnitude T.== 0) 1 magnitude
      factor = T.select (magnitude T.<= limit) 1 (limit / safeNorm)
      visit :: ValueTree GradientTensor fields -> fields -> fields
      visit TensorValues = scaleGradient factor
      visit (PairValues left right) = bimap (visit left) (visit right)
   in rebuild @Tensor $ visit (values @(Leaves Tensor p) @GradientTensor) $ flatten @Tensor gradients

-- | L1 norm of all gradient elements across every tensor field.
-- Supports single tensors, tuples and nested differentiable records with Float
-- or Double leaves. Accumulation and the scalar result use Double precision,
-- including for mixed-precision records. Empty tensor fields contribute zero.
gradientNormL1 ::
  (Differentiable p, ParametersConstraints GradientTensor p) =>
  p ->
  Tensor '[] Double
gradientNormL1 = foldGradients (reduceGradient T.normL1) (+)

-- | L2 norm of the concatenated gradient elements from every tensor field.
-- This is not the sum of the individual tensor norms. Float fields are promoted
-- to Double, and partial norms are combined with scaling to avoid intermediate
-- overflow and underflow. The result is a scalar Double tensor.
gradientNormL2 ::
  (Differentiable p, ParametersConstraints GradientTensor p) =>
  p ->
  Tensor '[] Double
gradientNormL2 =
  let combine left right =
        let x = T.index @'[] left
            y = T.index @'[] right
            peak = T.max x y
            divisor = T.select ((peak T.== 0) T.|| T.isInfinite peak) 1 peak
            a = x / divisor
            b = y / divisor
         in T.scalar $ T.select (T.isInfinite peak) peak (peak * sqrt (a * a + b * b))
   in foldGradients (reduceGradient T.normL2) combine

-- | Bound the global L1 norm by scaling all gradient fields together.
-- The caller supplies a finite, nonnegative maximum norm. Gradients below the
-- limit and all-zero gradients are preserved. Nonfinite gradients are not repaired.
-- Tensor dtypes and the containing record structure are preserved.
clipGradNormL1 ::
  (Differentiable p, ParametersConstraints GradientTensor p) =>
  Double ->
  p ->
  p
clipGradNormL1 = clipGradNorm gradientNormL1

-- | Bound the global L2 norm by scaling all gradient fields together.
-- The caller supplies a finite, nonnegative maximum norm. No host transfer or
-- optimizer update is performed. Zero gradients are preserved, including when
-- the limit is zero. Apply this after differentiation and before an optimizer step.
--
-- @
-- let (loss, gradients) = AG.backprop objective parameters
--     clipped = NN.clipGradNormL2 1 gradients
--     updated = NN.runState
--       (NN.optimizerStep optimizer parameters clipped)
--       optimizerState
-- @
clipGradNormL2 ::
  (Differentiable p, ParametersConstraints GradientTensor p) =>
  Double ->
  p ->
  p
clipGradNormL2 = clipGradNorm gradientNormL2
