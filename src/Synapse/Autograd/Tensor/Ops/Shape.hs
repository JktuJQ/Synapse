{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Shape transformations with pullbacks to the original tensor coordinates.
module Synapse.Autograd.Tensor.Ops.Shape () where

import qualified Data.Array.Accelerate as A
import GHC.TypeLits (natVal)
import Synapse.Autograd.Tensor.Helpers (axesOf, reindexGradient, reshapeLike, shapeOf, sumBroadcast, tensorBinary, tensorUnary)
import Synapse.Autograd.Tensor.Type (Tensor)
import Synapse.Autograd.Type (Context)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Shape (ShapeOps (..))
import qualified Synapse.Tensor.Type as T

instance (Context s, DType a, Num a, Num (T.TensorElement a)) => ShapeOps (Tensor s) a where
  reshapeOp = tensorUnary (\x -> (reshapeOp x, reshapeLike x))
  squeezeOp axis = tensorUnary (\x -> (squeezeOp axis x, reshapeLike x))
  unsqueezeOp axis = tensorUnary (\x -> (unsqueezeOp axis x, reshapeLike x))
  permuteAxesOp order = tensorUnary $ \x ->
    (permuteAxesOp order x, reindexGradient (\coordinates -> map (coordinates !!) (axesOf order)) x)
  transposeOp = tensorUnary $ \x ->
    let rank = length (shapeOf x)
        order = [0 .. rank - 3] ++ [rank - 1, rank - 2]
     in (transposeOp x, reindexGradient (\coordinates -> map (coordinates !!) order) x)
  concatOp selected = tensorBinary $ \x y ->
    let axis = fromIntegral (natVal selected)
        offset = shapeOf x !! axis
        shift = zipWith (\i coordinate -> if i == axis then coordinate + A.constant offset else coordinate) [0 ..]
     in (concatOp selected x y, \gradient -> (reindexGradient id x gradient, reindexGradient shift y gradient))
  stackOp selected = tensorBinary $ \x y ->
    let axis = fromIntegral (natVal selected)
        insert value coordinates = take axis coordinates ++ [value] ++ drop axis coordinates
     in (stackOp selected x y, \gradient -> (reindexGradient (insert 0) x gradient, reindexGradient (insert 1) y gradient))
  broadcastOp = tensorUnary (\x -> (broadcastOp x, sumBroadcast x))
