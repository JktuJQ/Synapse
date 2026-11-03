{-# LANGUAGE DataKinds #-}

module IntegralGradient where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[2] Int -> T.Tensor '[2] Int
invalid = NN.clipGradNormL2 1
