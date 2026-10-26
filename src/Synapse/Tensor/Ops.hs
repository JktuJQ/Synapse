{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}

-- | Lift tensor computations while preserving the behavior of their representation.
-- Ordinary tensors evaluate only the forward function.
-- Differentiable tensors register the supplied pullback without executing it during the forward pass.
module Synapse.Tensor.Ops (TensorOps (..)) where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (DType)
import qualified Synapse.Tensor.Type as T

-- | Tensor representations supporting primitive computations and gradient rules.
-- The first function computes the result; the second receives the original inputs, the forward result, and an output gradient.
-- Its result contains one input gradient per argument, in argument order and with the original input shapes.
-- Fractional dictionaries are supplied to pullbacks only when gradients are recorded.
-- Ordinary tensors also support integral and boolean elements; recording gradients requires fractional elements.
-- Inputs share an element type; the result may use a different element type.
-- Shape evidence is required only for the inputs whose gradient contributions may be accumulated.
class (DType a) => TensorOps (t :: T.TensorLike) a where
  -- | Embed a staged value without recording any incoming gradient edges.
  literal :: T.Tensor sh a -> t sh a

  -- | Read the staged value without its differentiation history.
  primal :: t sh a -> T.Tensor sh a

  -- | Convert an element expression into a scalar tensor, preserving its history.
  fromElement :: T.Element t a -> t '[] a

  -- | Lift an element-valued computation without introducing an intermediate scalar tensor.
  -- Element expressions can depend on coordinates local to a generator.
  liftElement ::
    (Shape sh) =>
    (T.Tensor sh a -> T.TensorElement b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh a -> T.TensorElement b -> T.TensorElement b -> T.Tensor sh a) ->
    t sh a ->
    T.Element t b

  -- | Lift a unary computation and its pullback.
  lift1 ::
    (Shape sh1) =>
    (T.Tensor sh1 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor out b -> T.Tensor out b -> T.Tensor sh1 a) ->
    t sh1 a ->
    t out b

  -- | Lift a 2-input computation and its pullback.
  lift2 ::
    (Shape sh1, Shape sh2) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a)) ->
    t sh1 a ->
    t sh2 a ->
    t out b

  -- | Lift a 3-input computation and its pullback.
  lift3 ::
    (Shape sh1, Shape sh2, Shape sh3) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a, T.Tensor sh3 a)) ->
    t sh1 a ->
    t sh2 a ->
    t sh3 a ->
    t out b

  -- | Lift a 4-input computation and its pullback.
  lift4 ::
    (Shape sh1, Shape sh2, Shape sh3, Shape sh4) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a, T.Tensor sh3 a, T.Tensor sh4 a)) ->
    t sh1 a ->
    t sh2 a ->
    t sh3 a ->
    t sh4 a ->
    t out b

  -- | Lift a 5-input computation and its pullback.
  lift5 ::
    (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a, T.Tensor sh3 a, T.Tensor sh4 a, T.Tensor sh5 a)) ->
    t sh1 a ->
    t sh2 a ->
    t sh3 a ->
    t sh4 a ->
    t sh5 a ->
    t out b

  -- | Lift a 6-input computation and its pullback.
  lift6 ::
    (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a, T.Tensor sh3 a, T.Tensor sh4 a, T.Tensor sh5 a, T.Tensor sh6 a)) ->
    t sh1 a ->
    t sh2 a ->
    t sh3 a ->
    t sh4 a ->
    t sh5 a ->
    t sh6 a ->
    t out b

  -- | Lift a 7-input computation and its pullback.
  lift7 ::
    (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6, Shape sh7) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor sh7 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor sh7 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a, T.Tensor sh3 a, T.Tensor sh4 a, T.Tensor sh5 a, T.Tensor sh6 a, T.Tensor sh7 a)) ->
    t sh1 a ->
    t sh2 a ->
    t sh3 a ->
    t sh4 a ->
    t sh5 a ->
    t sh6 a ->
    t sh7 a ->
    t out b

  -- | Lift a 8-input computation and its pullback.
  lift8 ::
    (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6, Shape sh7, Shape sh8) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor sh7 a -> T.Tensor sh8 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor sh7 a -> T.Tensor sh8 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a, T.Tensor sh3 a, T.Tensor sh4 a, T.Tensor sh5 a, T.Tensor sh6 a, T.Tensor sh7 a, T.Tensor sh8 a)) ->
    t sh1 a ->
    t sh2 a ->
    t sh3 a ->
    t sh4 a ->
    t sh5 a ->
    t sh6 a ->
    t sh7 a ->
    t sh8 a ->
    t out b

  -- | Lift a 9-input computation and its pullback.
  lift9 ::
    (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6, Shape sh7, Shape sh8, Shape sh9) =>
    (T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor sh7 a -> T.Tensor sh8 a -> T.Tensor sh9 a -> T.Tensor out b) ->
    ((Fractional a, Fractional (T.TensorElement a)) => T.Tensor sh1 a -> T.Tensor sh2 a -> T.Tensor sh3 a -> T.Tensor sh4 a -> T.Tensor sh5 a -> T.Tensor sh6 a -> T.Tensor sh7 a -> T.Tensor sh8 a -> T.Tensor sh9 a -> T.Tensor out b -> T.Tensor out b -> (T.Tensor sh1 a, T.Tensor sh2 a, T.Tensor sh3 a, T.Tensor sh4 a, T.Tensor sh5 a, T.Tensor sh6 a, T.Tensor sh7 a, T.Tensor sh8 a, T.Tensor sh9 a)) ->
    t sh1 a ->
    t sh2 a ->
    t sh3 a ->
    t sh4 a ->
    t sh5 a ->
    t sh6 a ->
    t sh7 a ->
    t sh8 a ->
    t sh9 a ->
    t out b

instance (DType a) => TensorOps T.Tensor a where
  literal = id
  primal = id
  fromElement (T.TensorElement x) = T.Tensor (A.unit x)
  liftElement forward _ = forward
  lift1 forward _ = forward
  lift2 forward _ = forward
  lift3 forward _ = forward
  lift4 forward _ = forward
  lift5 forward _ = forward
  lift6 forward _ = forward
  lift7 forward _ = forward
  lift8 forward _ = forward
  lift9 forward _ = forward
