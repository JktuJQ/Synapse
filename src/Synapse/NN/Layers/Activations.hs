{- | Provides activation functions - unary functions that are differentiable almost everywhere and so they can be used in backward loss propagation.
-}


module Synapse.NN.Layers.Activations
    ( -- * 'ActivationFn' type alias and 'Activation' newtype

      ActivationFn
    , activateScalar
    , activateMat

    , Activation (Activation, unActivation)
    , layerActivation

      -- * Activation functions

    , relu
    , sigmoid
    ) where


import Synapse.NN.Layers.Layer (AbstractLayer(..), LayerConfiguration)

import Synapse.Tensors (ElementwiseScalarOps((+.), (/.)), SingletonOps(unSingleton))

import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M

import Synapse.Autograd (Symbol(unSymbol), SymbolMat, Symbolic, constSymbol)


-- | 'ActivationFn' is a type alias that represents unary functions that differentiable almost everywhere.
type ActivationFn a = SymbolMat a -> SymbolMat a


-- | Applies activation function to a scalar to produce new scalar.
activateScalar :: Symbolic a => ActivationFn a -> a -> a
activateScalar fn = unSingleton . unSymbol . fn . constSymbol . M.singleton

-- | Applies activation function to a scalar to produce new scalar.
activateMat :: Symbolic a => ActivationFn a -> Mat a -> Mat a
activateMat fn = unSymbol . fn . constSymbol


{- | 'Activation' newtype wraps 'ActivationFn's - unary functions that can be thought of as activation functions for neural network layers.

Any activation function must be differentiable almost everywhere and so
it must be function that operates on 'Synapse.Autograd.Symbol's, which is allows for function to be differentiated when needed.
-}
newtype Activation a = Activation
    { unActivation :: ActivationFn a  -- ^ Unwraps 'Activation' newtype.
    }

instance AbstractLayer Activation where
    inputSize _ = Nothing
    outputSize _ = Nothing

    nParameters _ = 0
    getParameters _ _ = []
    updateParameters = const

    symbolicForward _ input (Activation fn) = (fn input, M.singleton 0)

-- | Creates configuration for activation layer.
layerActivation :: Activation a -> LayerConfiguration (Activation a)
layerActivation = const


-- Activation functions

-- | ReLU function.
relu :: (Symbolic a, Fractional a) => ActivationFn a
relu x = (x + abs x) /. 2.0

-- | Sigmoid function.
sigmoid :: (Symbolic a, Floating a) => ActivationFn a
sigmoid x = recip $ exp (negate x) +. 1.0
