{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

{- | Initialize tensors with uniformly or normally distributed values.

@
import Control.Monad.Trans.State.Strict (runState)
import System.Random (mkStdGen)
import qualified Synapse.NN.Initialization as I

let initialize = do
        weights <- I.xavierUniform \@'[2, 3] \@Double 2 3 1
        bias <- I.uniform \@'[3] \@Double (-0.1) 0.1
        pure (weights, bias)
    ((weights, bias), nextGenerator) = runState initialize (mkStdGen 42)
@

The same initial generator reproduces the same sequence of tensors and final generator.
An empty tensor or a constant distribution leaves the generator unchanged.

Xavier initializers take explicit input and output fan counts and a gain.
Kaiming initializers take one fan count and the negative-input slope of a leaky ReLU.
-}
module Synapse.NN.Initialization
    ( -- * Distributions
      uniform
    , normal

      -- * Xavier initialization
    , xavierUniform
    , xavierNormal

      -- * Kaiming initialization
    , kaimingUniform
    , kaimingNormal
    )
where

import Control.Monad.Trans.State.Strict (State)
import System.Random (RandomGen, UniformRange)
import System.Random.Stateful (StateGenM (..), uniformListRM, uniformRM)

import Synapse.Tensor.Data (unsafeFromList)
import Synapse.Tensor.Ops.Construction (full)
import Synapse.Tensor.Representation (DType, Tensor)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)

-- | Checks whether a supplied float value is finite.
finite
    :: RealFloat a
    => a
    -> Bool
finite x = not (isNaN x || isInfinite x)

{- | Initialize elements uniformly between the supplied lower and upper bounds.
Bounds must be finite and ordered; equal bounds produce a constant tensor.
Invalid bounds raise an error.
-}
uniform
    :: forall sh a g
     . (Shape sh, DType a, RealFloat a, UniformRange a, RandomGen g)
    => a
    -> a
    -> State g (Tensor sh a)
uniform lower upper
    | not (finite lower && finite upper) || lower > upper =
        error "Synapse.NN.uniform: bounds must be finite and ordered"
    | lower == upper = pure $ full @sh lower
    | otherwise =
        unsafeFromList @sh <$> uniformListRM (shapeSize @sh) (lower, upper) (StateGenM :: StateGenM g)

{- | Initialize elements from a normal distribution with the supplied mean and standard deviation.
Both parameters must be finite and the standard deviation must be nonnegative.
A zero standard deviation produces a constant tensor; invalid parameters raise an error.
-}
normal
    :: forall sh a g
     . (Shape sh, DType a, RealFloat a, RandomGen g)
    => a
    -> a
    -> State g (Tensor sh a)
normal mean std
    | not (finite mean && finite std) || std < 0 =
        error "Synapse.NN.normal: mean must be finite and standard deviation must be finite and nonnegative"
    | std == 0 = pure $ full @sh mean
    | otherwise = unsafeFromList @sh . map (\z -> mean + std * realToFrac z) <$> samples (shapeSize @sh) []
  where
    -- Marsaglia polar sampling consumes both draws even when a pair is rejected.
    samples 0 acc = pure $ reverse acc
    samples count acc = do
        u <- uniformRM (-1, 1) (StateGenM :: StateGenM g)
        v <- uniformRM (-1, 1) (StateGenM :: StateGenM g)
        let radius = u * u + v * v
        if radius <= 0 || radius >= 1
            then samples count acc
            else
                let scale = sqrt (((-2) * log radius) / radius)
                    first = u * scale :: Double
                    second = v * scale
                 in if count == 1
                        then pure $ reverse $ first : acc
                        else samples (count - 2) (second : first : acc)

-- | Compute the standard deviation for Xavier initialization.
xavierScale
    :: RealFloat a
    => Int
    -> Int
    -> a
    -> a
xavierScale fanIn fanOut gain
    | fanIn <= 0 || fanOut <= 0 = error "Synapse.NN.xavier: fanIn and fanOut must be positive"
    | not (finite gain) || gain < 0 = error "Synapse.NN.xavier: gain must be finite and nonnegative"
    | otherwise = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))

-- | Compute the standard deviation for Kaiming initialization.
kaimingScale
    :: RealFloat a
    => Int
    -> a
    -> a
kaimingScale fan slope
    | fan <= 0 = error "Synapse.NN.kaiming: fan must be positive"
    | not (finite slope) = error "Synapse.NN.kaiming: slope must be finite"
    | abs slope <= 1 = sqrt (2 / (1 + slope * slope)) / sqrt (fromIntegral fan)
    | otherwise = (sqrt (2 / (1 + recip slope * recip slope)) / abs slope) / sqrt (fromIntegral fan)

{- | Initialize uniformly on @[-bound, bound]@, where @bound = gain * sqrt (6 / (fanIn + fanOut))@.
Fan counts must be positive and gain must be finite and nonnegative.
Use a gain of one for linear or sigmoid activations.
-}
xavierUniform
    :: forall sh a g
     . (Shape sh, DType a, RealFloat a, UniformRange a, RandomGen g)
    => Int
    -> Int
    -> a
    -> State g (Tensor sh a)
xavierUniform fanIn fanOut gain =
    let bound = sqrt 3 * xavierScale fanIn fanOut gain
     in uniform @sh (-bound) bound

{- | Initialize normally with mean zero and @std = gain * sqrt (2 / (fanIn + fanOut))@.
Fan counts and gain have the same requirements as for 'xavierUniform'.
-}
xavierNormal
    :: forall sh a g
     . (Shape sh, DType a, RealFloat a, RandomGen g)
    => Int
    -> Int
    -> a
    -> State g (Tensor sh a)
xavierNormal fanIn fanOut gain = normal @sh 0 (xavierScale fanIn fanOut gain)

{- | Initialize uniformly on @[-bound, bound]@, where @bound = sqrt (6 / ((1 + slope²) * fan))@.
Supply fan-in to preserve forward variance or fan-out to preserve backward variance.
Fan must be positive and slope must be finite.
Use zero for ReLU or the negative-input slope for leaky ReLU.
-}
kaimingUniform
    :: forall sh a g
     . (Shape sh, DType a, RealFloat a, UniformRange a, RandomGen g)
    => Int
    -> a
    -> State g (Tensor sh a)
kaimingUniform fan slope =
    let bound = sqrt 3 * kaimingScale fan slope
     in uniform @sh (-bound) bound

{- | Initialize normally with mean zero and @std = sqrt (2 / ((1 + slope²) * fan))@.
Fan and slope have the same requirements as for 'kaimingUniform'.
-}
kaimingNormal
    :: forall sh a g
     . (Shape sh, DType a, RealFloat a, RandomGen g)
    => Int
    -> a
    -> State g (Tensor sh a)
kaimingNormal fan slope = normal @sh 0 (kaimingScale fan slope)
