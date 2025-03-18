{- | Allows to initialize values of layers parameters.

'InitializerFn' type alias represents functions that are able to initialize matrix with given size
and 'Initializer' newtype wraps 'InitializerFn's.

"Synapse" provides 4 types of initializers:
* Non-random constant initializers
* Random uniform distribution initializers
* Random normal distribution initializers
* Matrix-specific initializers
-}


module Synapse.NN.Layers.Initializers
    ( -- * 'InitializerFn' type alias and 'Initializer' newtype

      InitializerFn

    , Initializer (Initializer, unInitializer)
        
      -- * Non-random constant initializers

    , constants
    , zeroes
    , ones

      -- * Random uniform distribution initializers

    , randomUniform
    , lecunUniform
    , heUniform
    , glorotUniform

      -- * Random normal distribution initializers
    
    , randomNormal
    , lecunNormal
    , heNormal
    , glorotNormal

      -- * Matrix-like initializers
    
    , identity
    , orthogonal
    ) where


import Synapse.Tensors.Mat (Mat)
import qualified Synapse.Tensors.Mat as M

import System.Random (uniformListR, uniformRs, UniformRange, RandomGen)


-- | 'InitializerFn' type alias represents functions that are able to initialize matrix with given size.
type InitializerFn a = (Int, Int) -> Mat a


-- | 'Initializer' newtype wraps 'InitializerFn's - functions that are able to initialize matrix with given size.
newtype Initializer a = Initializer 
    { unInitializer :: InitializerFn a  -- ^ Unwraps 'Initializer' newtype.
    }


-- Non-random constant initializers

-- | Initializes list that is filled with given constant.
constants :: Num a => a -> InitializerFn a
constants c (input, output) = M.replicate (input, output) c

-- | Initializes list that is filled with zeroes.
zeroes :: Num a => InitializerFn a
zeroes = constants 0

-- | Initializes list that is filled with ones.
ones :: Num a => InitializerFn a
ones = constants 1


-- Random uniform distribution initializers

{- | Initializes list with samples from random uniform distribution in range.

This function does not preserve seed generator - split generator before calling this function.
-}
randomUniform :: (UniformRange a, RandomGen g) => (a, a) -> g -> InitializerFn a
randomUniform range gen sizes@(input, output) = M.fromList sizes $ fst $ uniformListR (input * output) range gen

{- | Initializes list with samples from random LeCun uniform distribution in range.

This function does not preserve seed generator - split generator before calling this function.
-}
lecunUniform :: (UniformRange a, Floating a, RandomGen g) => g -> InitializerFn a
lecunUniform gen sizes@(input, _) = let limit = sqrt $ 3.0 / fromIntegral input
                                    in randomUniform (-limit, limit) gen sizes

{- | Initializes list with samples from random He uniform distribution in range.

This function does not preserve seed generator - split generator before calling this function.
-}
heUniform :: (UniformRange a, Floating a, RandomGen g) => g -> InitializerFn a
heUniform gen sizes@(input, _) = let limit = sqrt $ 6.0 / fromIntegral input
                                 in randomUniform (-limit, limit) gen sizes

{- | Initializes list with samples from random Glorot uniform distribution in range.

This function does not preserve seed generator - split generator before calling this function.
-}
glorotUniform :: (UniformRange a, Floating a, RandomGen g) => g -> InitializerFn a
glorotUniform gen sizes@(input, output) = let limit = sqrt $ 6.0 / fromIntegral (input + output)
                                          in randomUniform (-limit, limit) gen sizes


-- Random normal distribution initializers

{- | Initializes list with samples from random normal distribution in range which could be truncated.

This function does not preserve seed generator - split generator before calling this function.
-}
randomNormal :: (UniformRange a, Floating a, Ord a, RandomGen g) => Maybe a -> a -> a -> g -> InitializerFn a
randomNormal truncated mean stdDev gen sizes@(input, output) = let us = pairs $ uniformRs (0.0, 1.0) gen
                                                                   ns = concatMap ((\(n1, n2) -> [n1, n2]) . transformBoxMuller) us
                                                                   ns' = map ((+ mean) . (* stdDev)) ns
                                                                   ns'' = case truncated of
                                                                               Nothing  -> ns'
                                                                               Just eps -> filter (\x -> abs (x - mean) < eps) ns'
                                                               in M.fromList sizes $ take (input * output) ns''
  where
    pairs [] = []
    pairs [x] = [(x, 1)]
    pairs (a:b:xs) = (a, b) : pairs xs

    transformBoxMuller (u1, u2) = let r = sqrt $ (-2.0) * log u1
                                      theta = 2.0 * pi * u2
                                  in (r * cos theta, r * sin theta)

{- | Initializes list with samples from random LeCun normal distribution in range
which is truncated for values more than two standard deviations from mean.

This function does not preserve seed generator - split generator before calling this function.
-}
lecunNormal :: (UniformRange a, Floating a, Ord a, RandomGen g) => g -> InitializerFn a
lecunNormal gen sizes@(input, _) = let mean = 0
                                       stdDev = sqrt $ 1.0 / fromIntegral input
                                   in randomNormal (Just $ 2.0 * stdDev) mean stdDev gen sizes

{- | Initializes list with samples from random He normal distribution in range
which is truncated for values more than two standard deviations from mean.

This function does not preserve seed generator - split generator before calling this function.
-}
heNormal :: (UniformRange a, Floating a, Ord a, RandomGen g) => g -> InitializerFn a
heNormal gen sizes@(input, _) = let mean = 0
                                    stdDev = sqrt $ 2.0 / fromIntegral input
                                in randomNormal (Just $ 2.0 * stdDev) mean stdDev gen sizes

{- | Initializes list with samples from random Glorot normal distribution in range
which is truncated for values more than two standard deviations from mean.

This function does not preserve seed generator - split generator before calling this function.
-}
glorotNormal :: (UniformRange a, Floating a, Ord a, RandomGen g) => g -> InitializerFn a
glorotNormal gen sizes@(input, output) = let mean = 0
                                             stdDev = sqrt $ 2.0 / fromIntegral (input + output)
                                         in randomNormal (Just $ 2.0 * stdDev) mean stdDev gen sizes


-- Matrix-like initializers

-- | Initializes flat identity matrix. If dimensions do not represent square matrix, an error is thrown.
identity :: Num a => InitializerFn a
identity (input, output)
    | input /= output = error "Given dimensions do not represent square matrix"
    | otherwise       = M.identity input

{- | Initializes float orthogonal matrix obtained from a random normal distribution
that is truncated for values more than two standard deviations from mean.

This function does not preserve seed generator - split generator before calling this function.
-}
orthogonal :: (UniformRange a, Floating a, Ord a, RandomGen g) => g -> InitializerFn a
orthogonal gen sizes = M.orthogonalized $ randomNormal Nothing 0.0 1.0 gen sizes
