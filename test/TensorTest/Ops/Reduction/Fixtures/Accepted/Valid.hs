{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Valid where

import qualified Synapse.Tensor as T

total :: T.TensorElement Int
total = T.sum (T.zeros @'[2, 3])

average :: T.TensorElement Double
average = T.mean (T.zeros @'[0])

scalarMinimum :: T.TensorElement Int
scalarMinimum = T.minimum (T.scalar 5)

columns :: T.Tensor '[3] Int
columns = T.sumAxes @'[2, 0] (T.zeros @'[2, 3, 4])

means :: T.Tensor '[2, 4] Float
means = T.meanAxes @'[1] (T.zeros @'[2, 3, 4])

emptyMaximum :: T.Tensor '[0] Int
emptyMaximum = T.maximumAxes @'[1] (T.zeros @'[0, 3])

unchangedMinimum :: T.Tensor '[0, 3] Int
unchangedMinimum = T.minimumAxes @'[] (T.zeros @'[0, 3])
