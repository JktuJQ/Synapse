{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Element where

import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T

bad :: T.TensorElement Double
bad = AG.gradient @(T.TensorElement Double) @(T.Tensor '[] Double) (const 1) (T.element 3)
