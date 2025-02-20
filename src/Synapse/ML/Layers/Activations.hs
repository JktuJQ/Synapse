{- | Provides activation functions - unary functions that are differentiable almost everywhere and so they can be used in backward loss propagation.
-}


{-# LANGUAGE DefaultSignatures         #-}  -- @DefaultSignatures@ are needed to provide default implementation in @ActivationFn@ typeclass.
{-# LANGUAGE ExistentialQuantification #-}  -- @ExistentialQuantification@ is needed to define @ActivationLayer@ datatype.


module Synapse.ML.Layers.Activations
    ( -- * @ActivationFn@ typeclass

      ActivationFn (callSymbolicMat, callScalar)
    , callFunctor

      -- * @ActivationLayer@ existential datatype
    
    , ActivationLayer (ActivationLayer)
    , activationLayer

      -- * Activation functions

    , Linear (Linear)

    , Sin (Sin)

    , Tanh (Tanh)

    , ReLU (ReLU, reluThreshold, reluLeftSlope, reluMaxValue)
    , defaultReLU
    ) where


import Synapse.ML.Layers.Layer (AbstractLayer(..), LayerConfiguration)

import Synapse.LinearAlgebra (unsafeIndex)

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Synapse.Autograd (Symbol(Symbol, unSymbol), Symbolic, constSymbol, symbolicUnaryOp)


{- | @ActivationFn@ typeclass describes unary functions that can be thought of as activation functions for neural network layers.

Any activation function should be differentiable almost everywhere and so
it must provide @callSymbolicMat@ function, which is allows for function to be differentiated when needed.
That function is very important, as it will be used in the backward loss propagation.

There is also @callScalar@ function that should allow cheap execution of function when gradients are not needed.
There is a default implementation that uses @callSymbolicMat@, but it is a bit inefficient - try to provide your own implementation.
That function is easily extended from scalars to functors over those scalars (see @callFunctor@).

@Synapse@ additionally requires so that activation functions could be serialized - that is to enhance user experience,
allowing to save and load any parts of your models. If you want to convert said parts to @String@, use @show . toJSON@.
-}
class Functor fn => ActivationFn fn where
    -- | Applies activation function to symbolic matrix to produce new symbolic matrix, while retaining gradients graph.
    callSymbolicMat :: (Symbolic a, Floating a, Ord a) => fn a -> Symbol (Mat a) -> Symbol (Mat a)

    -- | Applies activation function to a scalar to produce new scalar.
    callScalar :: (Floating a, Ord a) => fn a -> a -> a
    default callScalar :: (Symbolic a, Floating a, Ord a) => fn a -> a -> a
    callScalar fn x = unsafeIndex (unSymbol $ callSymbolicMat fn (constSymbol (M.singleton x))) (0, 0)

-- | Applies activation function to a functor to produce new functor.
callFunctor :: (ActivationFn fn, Functor f, Floating a, Ord a) => fn a -> f a -> f a
callFunctor fn = fmap (callScalar fn)


-- | @ActivationLayer@ existential datatype wraps anything that implements @ActivationFn@.
data ActivationLayer a = forall fn. ActivationFn fn => ActivationLayer (fn a)

instance Functor ActivationLayer where
    fmap f (ActivationLayer fn) = ActivationLayer $ fmap f fn

instance AbstractLayer ActivationLayer where
    inputSize _ = Nothing
    outputSize _ = Nothing

    getParameters _ = []
    updateParameters = const

    symbolicForward _ (ActivationLayer fn) = callSymbolicMat fn
    forward (ActivationLayer fn) = callFunctor fn


-- | Creates configuration for activation layer.
activationLayer :: ActivationFn fn => fn a -> LayerConfiguration (ActivationLayer a)
activationLayer fn = const $ ActivationLayer fn


-- Activation functions.

-- | Identity activation function.
data Linear a = Linear

instance Functor Linear where
    fmap _ _ = Linear

instance ActivationFn Linear where
    callSymbolicMat _ s = symbolicUnaryOp id s [(s, id)]
    callScalar _ = id


-- | Sinusoid activation function.
data Sin a = Sin

instance Functor Sin where
    fmap _ _ = Sin

instance ActivationFn Sin where
    callSymbolicMat _ = sin
    callScalar _ = sin


-- | Hyperbolic tangent activation function.
data Tanh a = Tanh

instance Functor Tanh where
    fmap _ _ = Tanh

instance ActivationFn Tanh where
    callSymbolicMat _ = tanh
    callScalar _ = tanh


-- | Rectified linear unit (ReLU) activation function.
data ReLU a = ReLU
    { reluThreshold :: a        -- ^ Defines threshold of ReLU function (values lower than @reluThreshold@ will be damped).
    , reluLeftSlope :: a        -- ^ Defines left slope of ReLU function (values lower than @reluThreshold@ will be multiplied by this coefficient).
    , reluMaxValue  :: Maybe a  -- ^ Sets maximum value of ReLU function (values greater than @reluMaxValue@ will be clamped to it).
    }

-- | Default version of @ReLU@ - threshold and left slope coefficient are set to 0 and no maximum clamping is done.
defaultReLU :: Num a => ReLU a
defaultReLU = ReLU 0 0 Nothing

instance Functor ReLU where
    fmap f (ReLU threshold leftSlope maxValue) = ReLU (f threshold) (f leftSlope) (fmap f maxValue)

instance ActivationFn ReLU where
    callSymbolicMat (ReLU threshold leftSlope Nothing) s@(Symbol _ mat _) =
        s * constSymbol (M.generate (M.size mat) (\i -> let x = unsafeIndex mat i
                                                        in if x < threshold then leftSlope else 1))
    callSymbolicMat (ReLU threshold leftSlope (Just maxValue)) s@(Symbol _ mat _) =
         s * constSymbol (M.generate (M.size mat) (\i -> let x = unsafeIndex mat i
                                                         in if x < threshold then leftSlope else 1))
           - constSymbol (M.generate (M.size mat) (\i -> let x = unsafeIndex mat i
                                                         in if x >= maxValue then x - maxValue else 0))

    callScalar (ReLU threshold leftSlope Nothing) x = if x < threshold then leftSlope * x else x
    callScalar (ReLU threshold leftSlope (Just maxValue)) x = let x' = callScalar (ReLU threshold leftSlope Nothing) x
                                                              in if x' > maxValue then maxValue else x
