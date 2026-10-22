{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Pullbacks for sums, means, extrema, and stable normalization.
module Synapse.Autograd.Tensor.Ops.Reduction () where

import Synapse.Autograd.Tensor.Helpers (axesOf, expandReduced, shapeOf, singletonAxis, tensorUnary)
import Synapse.Autograd.Tensor.Type (Tensor)
import Synapse.Autograd.Type (Context)
import qualified Synapse.Tensor.Construction as C
import Synapse.Tensor.DType (DType)
import qualified Synapse.Tensor.Ops.Elementwise as E
import Synapse.Tensor.Ops.Reduction (LogReductions (..), MeanReductions (..), Normalization (..), OrderedReductions (..), Reductions (..))
import qualified Synapse.Tensor.Type as T

instance (Context s, DType a, Num a, Num (T.TensorElement a)) => Reductions (Tensor s) a where
  sumOp = tensorUnary $ \x -> (sumOp x, expandReduced [0 .. length (shapeOf x) - 1] x)
  sumAxesOp selected = tensorUnary $ \x -> (sumAxesOp selected x, expandReduced (axesOf selected) x)

instance (Context s, DType a, RealFloat a, Fractional (T.TensorElement a)) => MeanReductions (Tensor s) a where
  meanOp = tensorUnary $ \x ->
    let count = T.constant (fromIntegral (product (shapeOf x)))
     in (meanOp x, E.map (/ count) . expandReduced [0 .. length (shapeOf x) - 1] x)
  meanAxesOp selected = tensorUnary $ \x ->
    let axes = axesOf selected
        count = T.constant (fromIntegral (product [dim | (axis, dim) <- zip [0 ..] (shapeOf x), axis `elem` axes]))
     in (meanAxesOp selected x, E.map (/ count) . expandReduced axes x)

-- | Equal extrema share the gradient uniformly, including reductions over several axes.
instance (Context s, DType a, Fractional a, Fractional (T.TensorElement a), E.Ord (T.TensorElement a)) => OrderedReductions (Tensor s) a where
  minimumOp = tensorUnary $ \x ->
    let y = minimumOp x
        axes = [0 .. length (shapeOf x) - 1]
        weights = E.zipWith (\a b -> E.select (a E.== b) 1 0) x (expandReduced axes x y)
        counts = sumOp weights
     in (y, \gradient -> weights * expandReduced axes x (gradient / counts))
  maximumOp = tensorUnary $ \x ->
    let y = maximumOp x
        axes = [0 .. length (shapeOf x) - 1]
        weights = E.zipWith (\a b -> E.select (a E.== b) 1 0) x (expandReduced axes x y)
        counts = sumOp weights
     in (y, \gradient -> weights * expandReduced axes x (gradient / counts))
  minimumAxesOp selected = tensorUnary $ \x ->
    let y = minimumAxesOp selected x
        axes = axesOf selected
        weights = E.zipWith (\a b -> E.select (a E.== b) 1 0) x (expandReduced axes x y)
        counts = sumAxesOp selected weights
     in (y, \gradient -> weights * expandReduced axes x (gradient / counts))
  maximumAxesOp selected = tensorUnary $ \x ->
    let y = maximumAxesOp selected x
        axes = axesOf selected
        weights = E.zipWith (\a b -> E.select (a E.== b) 1 0) x (expandReduced axes x y)
        counts = sumAxesOp selected weights
     in (y, \gradient -> weights * expandReduced axes x (gradient / counts))
  argmaxOp selected = tensorUnary $ \x -> (argmaxOp selected x, const (C.zerosLike x))
  argminOp selected = tensorUnary $ \x -> (argminOp selected x, const (C.zerosLike x))

instance (Context s, DType a, Floating a, Floating (T.TensorElement a), E.Ord (T.TensorElement a)) => Normalization (Tensor s) a where
  softmaxOp selected = tensorUnary $ \x ->
    let y = softmaxOp selected x
     in (y, \gradient -> y * (gradient - expandReduced (axesOf (singletonAxis selected)) x (sumAxesOp (singletonAxis selected) (gradient * y))))
  logSoftmaxOp selected = tensorUnary $ \x ->
    let y = logSoftmaxOp selected x
     in (y, \gradient -> gradient - exp y * expandReduced (axesOf (singletonAxis selected)) x (sumAxesOp (singletonAxis selected) gradient))

instance (Context s, DType a, RealFloat a, Floating (T.TensorElement a), E.Ord (T.TensorElement a), E.RealFloat (T.TensorElement a)) => LogReductions (Tensor s) a where
  logSumExpOp = tensorUnary $ \x ->
    let y = logSumExpOp x
        axes = [0 .. length (shapeOf x) - 1]
     in (y, \gradient -> exp (x - expandReduced axes x y) * expandReduced axes x gradient)
  logSumExpAxesOp selected = tensorUnary $ \x ->
    let y = logSumExpAxesOp selected x
        axes = axesOf selected
     in (y, \gradient -> exp (x - expandReduced axes x y) * expandReduced axes x gradient)
