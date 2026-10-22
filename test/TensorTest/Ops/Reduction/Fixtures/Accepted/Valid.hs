{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Valid where

import qualified Synapse.Tensor as T

total :: T.Tensor '[] Int
total = T.sum (T.zeros @'[2, 3])

average :: T.Tensor '[] Double
average = T.mean (T.zeros @'[0])

scalarMinimum :: T.Tensor '[] Int
scalarMinimum = T.minimum (T.scalar 5)

columns :: T.Tensor '[3] Int
columns = T.sumAxes @'[2, 0] (T.zeros @'[2, 3, 4])

means :: T.Tensor '[2, 4] Float
means = T.meanAxes @'[1] (T.zeros @'[2, 3, 4])

emptyMaximum :: T.Tensor '[0] Int
emptyMaximum = T.maximumAxes @'[1] (T.zeros @'[0, 3])

unchangedMinimum :: T.Tensor '[0, 3] Int
unchangedMinimum = T.minimumAxes @'[] (T.zeros @'[0, 3])

probabilities :: T.Tensor '[2, 3] Float
probabilities = T.softmax @1 (T.zeros @'[2, 3])

logProbabilities :: T.Tensor '[2, 3] Double
logProbabilities = T.logSoftmax @0 (T.zeros @'[2, 3])

logTotals :: T.Tensor '[3] Double
logTotals = T.logSumExpAxes @'[2, 0] (T.zeros @'[2, 3, 4])

emptyLogTotals :: T.Tensor '[2] Float
emptyLogTotals = T.logSumExpAxes @'[1] (T.zeros @'[2, 0])

booleanRows :: T.Tensor '[2] Bool
booleanRows = T.allAxes @'[1] (T.full @'[2, 0] True)

positions :: T.Tensor '[2, 4] Int
positions = T.argmax @1 (T.zeros @'[2, 3, 4] @Float)

minimumPosition :: T.Tensor '[] Int
minimumPosition = T.argmin @0 (T.zeros @'[3] @Int)

emptyPositions :: T.Tensor '[0] Int
emptyPositions = T.argmax @1 (T.zeros @'[0, 3] @Double)
