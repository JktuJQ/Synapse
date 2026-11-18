{-# LANGUAGE DataKinds #-}

module MixedGradient where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalidNorm :: (T.Tensor '[2] Float, T.Tensor '[1] Double) -> T.Tensor '[] Double
invalidNorm = NN.gradientNormL2

invalidClip :: (T.Tensor '[2] Float, T.Tensor '[1] Double) -> (T.Tensor '[2] Float, T.Tensor '[1] Double)
invalidClip = NN.clipGradNormL2 (1 :: Double)
