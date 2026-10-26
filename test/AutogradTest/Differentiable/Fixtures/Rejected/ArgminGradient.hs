{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ArgminGradient where

import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T

bad :: T.Tensor '[2, 3] Double -> (T.Tensor '[2] Int, T.Tensor '[2, 3] Double)
bad = AG.backprop (T.argmin @1)
