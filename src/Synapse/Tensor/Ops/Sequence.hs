{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Differentiable fixed-length sequence traversal.
module Synapse.Tensor.Ops.Sequence (mapAccum) where

import Control.Monad.Trans.State.Strict (State, runState, state)
import GHC.TypeNats (KnownNat)
import qualified Synapse.Autograd.Execution as AG
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Sequence as Reverse
import Synapse.Autograd.Type (Differentiable)
import Synapse.Tensor.Fields (Flatten, Leaves)
import Synapse.Tensor.Ops (Compatible, Needs (Fractional), TensorOps (..))
import qualified Synapse.Tensor.Ops.Primitives.Sequence as Raw
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor, TensorElement, WithTensor)

-- | Apply a stateful step from left to right along the leading sequence axis.
-- Parameters are shared by all steps. The result contains every step output;
-- the final state is returned through 'State'. An empty sequence preserves the
-- initial state and produces zero parameter gradients.
--
-- The backend uses a loop rather than expanding one graph per step. During
-- differentiation it records state histories and recomputes each local step
-- for its pullback. Ordinary evaluation does not retain those histories.
-- Differentiable parameters must be passed explicitly rather than captured by
-- the step function. Shapes and the structure of the state remain fixed.
-- A sequence of batched feature vectors has shape @'[time, batch, features]@;
-- each step receives a tensor of shape @'[batch, features]@.
--
-- @
-- import Control.Monad.Trans.State.Strict (runState, state)
--
-- step weight input = state $ \total ->
--   let next = total + weight * input
--   in (next, next)
--
-- -- Weighted running totals [2, 6, 12] and final state 12.
-- result = runState
--   (T.mapAccum step (T.full \@'[] (2 :: Double)) (T.fromList \@'[3] [1, 2, 3]))
--   (T.full \@'[] 0)
-- @
mapAccum ::
  forall n t a parameters recurrent input output.
  ( TensorOps t a,
    KnownNat n,
    Floating a,
    Floating (TensorElement a),
    WithTensor Tensor parameters ~ parameters,
    WithTensor Tensor recurrent ~ recurrent,
    Flatten Tensor parameters,
    Flatten Tensor recurrent,
    Shape input,
    Shape output,
    Shape (n ': input),
    Shape (n ': output),
    Differentiable (parameters, (Tensor input a, recurrent)),
    Differentiable (Tensor output a, recurrent),
    Raw.ScanValues a (Leaves Tensor parameters),
    Raw.ScanValues a (Leaves Tensor recurrent),
    Compatible t a (parameters, (Tensor (n ': input) a, recurrent)) (Tensor (n ': output) a, recurrent)
  ) =>
  (forall u. (TensorOps u a, forall sh. (Shape sh) => Floating (u sh a)) => WithTensor u parameters -> u input a -> State (WithTensor u recurrent) (u output a)) ->
  WithTensor t parameters ->
  t (n ': input) a ->
  State (WithTensor t recurrent) (t (n ': output) a)
mapAccum step parameters inputs = state $ \initial ->
  customVJP
    Fractional
    (Raw.forward @n stepForward)
    (\arguments -> let (output, history) = Raw.forwardCached @n stepForward arguments in (output, (arguments, history)))
    (Reverse.backward @n stepBackward)
    (parameters, (inputs, initial))
  where
    stepForward p x = runState $ step p x
    stepBackward arguments = snd $ AG.pullback (\(p, (x, s)) -> runState (step p x) s) arguments
