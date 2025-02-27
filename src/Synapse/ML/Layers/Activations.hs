{- | Provides activation functions - unary functions that are differentiable almost everywhere and so they can be used in backward loss propagation.
-}


module Synapse.ML.Layers.Activations
    ( -- * @ActivationFn@ type alias and @Activation@ newtype

      ActivationFn
    , activateScalar
    , activateMat

    , Activation (Activation, unActivation)
    , layerActivation

      -- * Activation functions

    , reluWith
    , relu
    ) where


import Synapse.ML.Layers.Layer (AbstractLayer(..), LayerConfiguration)

import Synapse.LinearAlgebra (Indexable(unsafeIndex), SingletonOps(unSingleton))

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Synapse.Autograd (Symbol(unSymbol), Symbolic, constSymbol)


-- | @ActivationFn@ is a type alias that represents unary functions that differentiable almost everywhere.
type ActivationFn a = Symbol (Mat a) -> Symbol (Mat a)

-- | Applies activation function to a scalar to produce new scalar.
activateScalar :: Symbolic a => ActivationFn a -> a -> a
activateScalar fn = unSingleton . unSymbol . fn . constSymbol . M.singleton

-- | Applies activation function to a scalar to produce new scalar.
activateMat :: Symbolic a => ActivationFn a -> Mat a -> Mat a
activateMat fn = unSymbol . fn . constSymbol


{- | @Activation@ newtype wraps @ActivationFn@s - unary functions that can be thought of as activation functions for neural network layers.

Any activation function must be differentiable almost everywhere and so
it must be function that operates on @Symbol@s, which is allows for function to be differentiated when needed.
-}
newtype Activation a = Activation 
    { unActivation :: ActivationFn a  -- ^ Unwraps @Activation@ newtype.
    }

instance AbstractLayer Activation where
    inputSize _ = Nothing
    outputSize _ = Nothing

    getParameters _ = []
    updateParameters = const

    applyRegularizer _ _ = constSymbol $ M.singleton 0

    symbolicForward _ (Activation fn) = fn

-- | Creates configuration for activation layer.
layerActivation :: ActivationFn a -> LayerConfiguration (Activation a)
layerActivation fn = const $ Activation fn


-- Activation functions

-- | Configurable ReLU function.
reluWith 
  :: (Symbolic a, Ord a)
  => a        -- ^ Threshold of ReLU function.
  -> a        -- ^ Left slope coefficient - slope of ReLU function on the left of threshold.
  -> Maybe a  -- ^ Maximum value clamping - all values greater than this will be clamped.
  -> ActivationFn a
reluWith threshold leftSlope Nothing s =
    s * constSymbol (M.generate (M.size $ unSymbol s) (\i -> let x = unsafeIndex (unSymbol s) i
                                                             in if x < threshold then leftSlope else 1))
reluWith threshold leftSlope (Just maxValue) s =
    s * constSymbol (M.generate (M.size $ unSymbol s) (\i -> let x = unsafeIndex (unSymbol s) i
                                                             in if x < threshold then leftSlope else 1))
    - constSymbol (M.generate (M.size $ unSymbol s) (\i -> let x = unsafeIndex (unSymbol s) i
                                                           in if x >= maxValue then x - maxValue else 0))

-- | Default version of ReLU - threshold and left slope coefficient are set to 0 and no maximum clamping is done.
relu :: (Symbolic a, Ord a) => ActivationFn a
relu = reluWith 0 0 Nothing
