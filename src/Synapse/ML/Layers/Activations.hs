{- | Provides activation functions - unary functions that are differentiable almost everywhere and so they can be used in backward loss propagation.
-}


{-# LANGUAGE DefaultSignatures      #-}  -- @DefaultSignatures@ are needed to provide default implementation in @ActivationFn@ typeclass.
{-# LANGUAGE DeriveGeneric          #-}  -- @DeriveGeneric@ are needed to easily derive @Serialize@ for activation functions.
{-# LANGUAGE FlexibleInstances      #-}  -- @FlexibleInstances@ are needed to implement @ActivationFn@ typeclass.
{-# LANGUAGE FunctionalDependencies #-}  -- @FunctionalDependencies@ are needed to implement @ActivationFn@ typeclass.


module Synapse.ML.Layers.Activations
    ( -- * @ActivationFn@ typeclass

      ActivationFn (callSymbolicMat, callScalar)
    , callFunctor
    
      -- * Activation functions

    , Linear

    , Sin

    , Tanh

    , ReLU (ReLU, reluThreshold, reluLeftSlope, reluMaxValue)
    , defaultReLU
    ) where


import Synapse.LinearAlgebra (unsafeIndex)

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Synapse.Autograd (Symbol(Symbol, unSymbol), Symbolic, constSymbol, symbolicUnaryOp)

import GHC.Generics (Generic)

import Data.Aeson (FromJSON, ToJSON)


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
class (FromJSON fn, ToJSON fn) => ActivationFn fn a | fn -> a where
    -- | Applies activation function to symbolic matrix to produce new symbolic matrix, while retaining gradients graph.
    callSymbolicMat :: Symbolic a => fn -> Symbol (Mat a) -> Symbol (Mat a)

    -- | Applies activation function to a scalar to produce new scalar.
    callScalar :: fn -> a -> a
    default callScalar :: Symbolic a => fn -> a -> a
    callScalar fn x = unsafeIndex (unSymbol $ callSymbolicMat fn (constSymbol (M.singleton x))) (0, 0)

-- | Applies activation function to a functor to produce new functor.
callFunctor :: (ActivationFn fn a , Functor f) => fn -> f a -> f a
callFunctor fn = fmap (callScalar fn)


-- | Identity activation function.
data Linear a = Linear
    {
    } deriving Generic

instance FromJSON (Linear a)
instance ToJSON (Linear a)

instance ActivationFn (Linear a) a where
    callSymbolicMat _ s = symbolicUnaryOp id s [(s, id)]
    callScalar _ = id


-- | Sinusoid activation function.
data Sin a = Sin
    {
    } deriving Generic

instance FromJSON (Sin a)
instance ToJSON (Sin a)

instance Floating a => ActivationFn (Sin a) a where
    callSymbolicMat _ = sin
    callScalar _ = sin


-- | Hyperbolic tangent activation function.
data Tanh a = Tanh
    {
    } deriving Generic

instance FromJSON (Tanh a)
instance ToJSON (Tanh a)

instance Floating a => ActivationFn (Tanh a) a where
    callSymbolicMat _ = tanh
    callScalar _ = tanh


-- | Rectified linear unit (ReLU) activation function.
data ReLU a = ReLU
    { reluThreshold :: a        -- ^ Defines threshold of ReLU function (values lower than @reluThreshold@ will be damped).
    , reluLeftSlope :: a        -- ^ Defines left slope of ReLU function (values lower than @reluThreshold@ will be multiplied by this coefficient).
    , reluMaxValue  :: Maybe a  -- ^ Sets maximum value of ReLU function (values greater than @reluMaxValue@ will be clamped to it).
    } deriving Generic

-- | Default version of @ReLU@ - threshold and left slope coefficient are set to 0 and no maximum clamping is done.
defaultReLU :: Num a => ReLU a
defaultReLU = ReLU 0 0 Nothing

instance FromJSON a => FromJSON (ReLU a)
instance ToJSON a => ToJSON (ReLU a)

instance (Num a, Ord a, FromJSON a, ToJSON a) => ActivationFn (ReLU a) a where
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
