{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Pure tensor initialization with explicit random generators.
-- The same parameters and generator reproduce the same tensor.
-- Split generators with <https://hackage.haskell.org/package/random/docs/System-Random.html#v:splitGen splitGen> before initializing independent parameters.
-- Fan counts are supplied explicitly and need not match dimensions of the result.
module Synapse.NN.Initialization
  ( uniform,
    normal,
    xavierUniform,
    xavierNormal,
    kaimingUniform,
    kaimingNormal,
  )
where

import Synapse.Tensor.Construction (full)
import Synapse.Tensor.Conversion (unsafeFromList)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Type (Tensor)
import System.Random (RandomGen, UniformRange, uniformR, uniformRs)

finite :: (RealFloat a) => a -> Bool
finite x = not (isNaN x || isInfinite x)

-- | Initialize elements uniformly between the supplied lower and upper bounds.
-- Bounds must be finite and ordered; equal bounds produce a constant tensor.
-- Invalid parameters raise an error.
uniform ::
  forall sh a g.
  (Shape sh, DType a, RealFloat a, UniformRange a, RandomGen g) =>
  a ->
  a ->
  g ->
  Tensor sh a
uniform lower upper generator
  | not (finite lower && finite upper) || lower > upper = error "Synapse.NN.uniform: bounds must be finite and ordered"
  | lower == upper = full @sh lower
  | otherwise = unsafeFromList @sh $ take (shapeSize @sh) $ uniformRs (lower, upper) generator

-- | Generate standard normal pairs using the Marsaglia polar method.
-- Rejected pairs exclude the origin and the boundary of the unit disk.
normalSamples :: (RandomGen g) => g -> [Double]
normalSamples generator =
  let (u, next) = uniformR (-1, 1) generator
      (v, rest) = uniformR (-1, 1) next
      radius = u * u + v * v
   in if radius <= 0 || radius >= 1
        then normalSamples rest
        else
          let scale = sqrt (((-2) * log radius) / radius)
           in u * scale : v * scale : normalSamples rest

-- | Initialize elements from a normal distribution with the supplied mean and standard deviation.
-- Both parameters must be finite and the standard deviation must be nonnegative.
-- A zero standard deviation produces a constant tensor; invalid parameters raise an error.
normal ::
  forall sh a g.
  (Shape sh, DType a, RealFloat a, RandomGen g) =>
  a ->
  a ->
  g ->
  Tensor sh a
normal mean std generator
  | not (finite mean && finite std) || std < 0 = error "Synapse.NN.normal: mean must be finite and standard deviation must be finite and nonnegative"
  | std == 0 = full @sh mean
  | otherwise = unsafeFromList @sh $ take (shapeSize @sh) $ map (\z -> mean + std * realToFrac z) $ normalSamples generator

xavierScale :: (RealFloat a) => Int -> Int -> a -> a
xavierScale fanIn fanOut gain
  | fanIn <= 0 || fanOut <= 0 = error "Synapse.NN.xavier: fanIn and fanOut must be positive"
  | not (finite gain) || gain < 0 = error "Synapse.NN.xavier: gain must be finite and nonnegative"
  | otherwise = gain * sqrt (2 / (fromIntegral fanIn + fromIntegral fanOut))

kaimingScale :: (RealFloat a) => Int -> a -> a
kaimingScale fan slope
  | fan <= 0 = error "Synapse.NN.kaiming: fan must be positive"
  | not (finite slope) = error "Synapse.NN.kaiming: slope must be finite"
  | abs slope <= 1 = sqrt (2 / (1 + slope * slope)) / sqrt (fromIntegral fan)
  | otherwise = (sqrt (2 / (1 + recip slope * recip slope)) / abs slope) / sqrt (fromIntegral fan)

-- | Apply Xavier initialization with a uniform distribution on @[-bound, bound]@.
-- Arguments are @fanIn@, @fanOut@, @gain@, then the generator, with @bound = gain * sqrt (6 / (fanIn + fanOut))@.
-- Fan counts must be positive and gain must be finite and nonnegative.
-- Use a gain of one for linear and sigmoid activations.
xavierUniform ::
  forall sh a g.
  (Shape sh, DType a, RealFloat a, UniformRange a, RandomGen g) =>
  Int ->
  Int ->
  a ->
  g ->
  Tensor sh a
xavierUniform fanIn fanOut gain =
  let bound = sqrt 3 * xavierScale fanIn fanOut gain
   in uniform @sh (-bound) bound

-- | Apply Xavier initialization with mean zero and @std = gain * sqrt (2 / (fanIn + fanOut))@.
-- Arguments and parameter requirements are the same as for 'xavierUniform'.
xavierNormal ::
  forall sh a g.
  (Shape sh, DType a, RealFloat a, RandomGen g) =>
  Int ->
  Int ->
  a ->
  g ->
  Tensor sh a
xavierNormal fanIn fanOut gain = normal @sh 0 (xavierScale fanIn fanOut gain)

-- | Apply Kaiming initialization with a uniform distribution on @[-bound, bound]@.
-- Arguments are @fan@, @slope@, then the generator, with @bound = sqrt (6 / ((1 + slope²) * fan))@.
-- Supply fan-in to preserve forward variance or fan-out to preserve backward variance.
-- Fan must be positive and slope must be finite; use zero for ReLU or the negative-input slope for leaky ReLU.
kaimingUniform ::
  forall sh a g.
  (Shape sh, DType a, RealFloat a, UniformRange a, RandomGen g) =>
  Int ->
  a ->
  g ->
  Tensor sh a
kaimingUniform fan slope =
  let bound = sqrt 3 * kaimingScale fan slope
   in uniform @sh (-bound) bound

-- | Apply Kaiming initialization with mean zero and @std = sqrt (2 / ((1 + slope²) * fan))@.
-- Arguments and parameter requirements are the same as for 'kaimingUniform'.
kaimingNormal ::
  forall sh a g.
  (Shape sh, DType a, RealFloat a, RandomGen g) =>
  Int ->
  a ->
  g ->
  Tensor sh a
kaimingNormal fan slope = normal @sh 0 (kaimingScale fan slope)
