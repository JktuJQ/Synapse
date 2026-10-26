{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Number where

import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T

bad :: Double
bad = AG.gradient @Double @(T.Tensor '[] Double) (const 1) 3
