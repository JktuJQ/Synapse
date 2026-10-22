{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Linear where

import qualified Synapse.Tensor as T

matrix :: T.Tensor '[2, 4] Int
matrix = T.matmul (T.ones @'[2, 3]) (T.ones @'[3, 4])

batched :: T.Tensor '[2, 8, 3, 5] Float
batched = T.matmul (T.ones @'[2, 8, 3, 4]) (T.ones @'[2, 8, 4, 5])

inner :: T.Tensor '[] Double
inner = T.dot (T.ones @'[4]) (T.ones @'[4])

empty :: T.Tensor '[2, 3] Int
empty = T.matmul (T.ones @'[2, 0]) (T.ones @'[0, 3])

expanded :: T.Tensor '[8, 2, 4] Float
expanded = T.matmul (T.broadcast @'[8, 2, 3] (T.ones @'[2, 3])) (T.ones @'[8, 3, 4])

project :: (T.LinearAlgebra t Double) => t '[2, 3] Double -> t '[3, 4] Double -> t '[2, 4] Double
project = T.matmul

squaredTotal :: (T.Reductions t Double, Num (t '[2, 3] Double)) => t '[2, 3] Double -> t '[] Double
squaredTotal input = T.sum (input * input)

reshapeMatrix :: (T.ShapeOps t Double) => t '[2, 3] Double -> t '[3, 2] Double
reshapeMatrix = T.reshape @'[3, 2]

convertElements :: (T.Elementwise t, T.FromIntegral (T.Element t Int) (T.Element t Double)) => t '[3] Int -> t '[3] Double
convertElements = T.map T.fromIntegral

generateCoordinates :: (T.Construction t, T.Coordinates (T.Element t), Num (T.Element t Int)) => t '[2, 3] Int
generateCoordinates = T.fromGen @'[2, 3] (\ix -> T.coordinate @0 ix * 10 + T.coordinate @1 ix)

readElement :: (T.Indexing t Double) => t '[2, 3] Double -> T.Element t Double
readElement = T.index @'[1, 2]

selectRow :: (T.Slicing t Double) => t '[2, 3] Double -> t '[3] Double
selectRow = T.slice @'[ 'T.At 1, 'T.All]

makeScalar :: (T.Construction t) => T.Element t Double -> t '[] Double
makeScalar = T.scalar

swapPair :: (T.ElementControl element) => element (Double, Bool) -> element (Bool, Double)
swapPair input = T.pair (T.snd input) (T.fst input)
