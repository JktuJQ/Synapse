{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Dropout with explicit random state or caller-supplied masks.
module Synapse.NN.Layers.Dropout
  ( DropoutState (..),
    initializeDropout,
    dropout,
    dropoutWithMask,
  )
where

import Control.Arrow (Arrow, arr)
import GHC.Generics (Generic)
import Synapse.NN.Execution (ExecutionMode (..))
import Synapse.Tensor.Ops.Elementwise (Conditional (..))
import Synapse.Tensor.Random (Generator, bernoulli, initializeGenerator)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor, WithTensor)
import System.Random (RandomGen)

-- | Random state for one dropout operation, independent of trainable parameters.
-- Initialize separate states for independently sampled dropout operations.
-- The state can be a field of a user-defined record passed through a prepared computation.
newtype DropoutState t = DropoutState
  { dropoutGenerator :: Generator t
  }
  deriving (Generic)

type instance WithTensor t (DropoutState Tensor) = DropoutState t

-- | Initialize dropout from a host random generator and return the updated host generator.
-- For an explicit seed, use @DropoutState (T.generatorFromSeed seed)@ instead.
initializeDropout :: (RandomGen g) => g -> (DropoutState Tensor, g)
initializeDropout generator =
  let (randomState, next) = initializeGenerator generator
   in (DropoutState randomState, next)

-- | Retain elements selected by a mask and scale them by @1 / (1 - probability)@.
-- The caller supplies a probability in @[0, 1]@ and a mask whose entries are independently true with probability @1 - probability@.
-- Evaluation returns the input without inspecting the probability or mask.
-- Probabilities zero and one return the input and zeros respectively without inspecting the mask.
dropoutWithMask :: forall {arrow} a. (Arrow arrow, Fractional a, Conditional a) => ExecutionMode -> Double -> Condition a -> arrow a a
dropoutWithMask Eval _ _ = arr id
dropoutWithMask Train probability mask
  | probability == 0 = arr id
  | probability == 1 = arr $ const 0
  | otherwise = arr $ \input -> select mask (input / realToFrac (1 - probability)) 0

-- | Independently zero each element with the given probability during training.
-- Retained elements are scaled by @1 / (1 - probability)@ to preserve their expected value.
-- The caller supplies a finite probability in @[0, 1]@.
-- Returns the output and the random state for the next call.
-- Gradients flow through retained input elements; the random mask has no differentiation history.
-- Evaluation returns the input and unchanged state without inspecting the probability or state.
-- Probabilities zero and one return the input and zeros respectively without advancing the state.
dropout :: forall sh t a {arrow}. (Arrow arrow, Shape sh, Fractional (t sh a), Conditional (t sh a), Condition (t sh a) ~ Tensor sh Bool) => ExecutionMode -> Double -> DropoutState Tensor -> arrow (t sh a) (t sh a, DropoutState Tensor)
dropout Eval _ state = arr (,state)
dropout Train probability state
  | probability == 0 = arr (,state)
  | probability == 1 = arr $ const (0, state)
  | otherwise = arr $ \input ->
      let (mask, next) = bernoulli @sh (1 - probability) $ dropoutGenerator state
       in (dropoutWithMask Train probability mask input, DropoutState next)
