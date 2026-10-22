{-# LANGUAGE DataKinds #-}

module AutogradTest.Execution.Public (squaredGradient, constantGradient) where

import qualified Synapse.Autograd.Execution as AG
import Synapse.Autograd.Tensor (Tensor)
import qualified Synapse.Tensor as T

squaredGradient :: T.Tensor '[3] Double -> T.Tensor '[3] Double
squaredGradient = AG.gradient (\x -> x * x)

constantTensor :: T.Tensor '[3] Double -> Tensor s '[3] Double
constantTensor = AG.constant

constantGradient :: T.Tensor '[3] Double -> T.Tensor '[3] Double -> T.Tensor '[3] Double
constantGradient frozen = AG.gradient (\x -> x * constantTensor frozen)
