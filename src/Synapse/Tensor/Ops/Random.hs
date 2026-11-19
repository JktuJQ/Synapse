{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Random tensor sampling with explicit generator state.
module Synapse.Tensor.Ops.Random
    ( -- * Generator state
      StagedGenerator
    , generatorFromSeed
    , initializeGenerator

      -- * Sampling actions
    , randomWords
    , uniform
    , bernoulli

      -- * Counter-based generation
    , philox
    ) where

import Control.Monad.Trans.State.Strict (State)
import Data.Word (Word32)

import Synapse.Tensor.Ops.Constraints (FractionalDType)
import Synapse.Tensor.Ops.Conversion (FromIntegral)
import Synapse.Tensor.Ops.Raw.Random (StagedGenerator, generatorFromSeed, initializeGenerator, philox)
import Synapse.Tensor.Representation (Tensor, TensorElement)
import Synapse.Tensor.Shapes.Dimensions (Shape)

import qualified Synapse.Tensor.Ops.Raw.Random as Raw

-- | Sample uniformly distributed 32-bit words and advance the generator.
randomWords
    :: forall sh
     . Shape sh
    => State (StagedGenerator Tensor) (Tensor sh Word32)
randomWords = Raw.randomWords @sh

-- | Sample uniformly from @[0, 1)@ and advance the generator.
uniform
    :: forall sh a
     . (Shape sh, RealFloat a, FractionalDType a, FromIntegral (TensorElement Word32) (TensorElement a))
    => State (StagedGenerator Tensor) (Tensor sh a)
uniform = Raw.uniform @sh @a

-- | Sample independent boolean values with the supplied probability of being true.
bernoulli
    :: forall sh
     . Shape sh
    => Double
    -> State (StagedGenerator Tensor) (Tensor sh Bool)
bernoulli = Raw.bernoulli @sh
