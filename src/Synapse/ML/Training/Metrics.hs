{- | Provides collection of functions that are used to judge the performance of models.

@MetricFn@ type alias represents those functions, and @Synapse@ offers a variety of them.
-}


module Synapse.ML.Training.Metrics
    ( -- @MetricFn@ type alias

      MetricFn

    , lossToMetric
    ) where


import Synapse.ML.Training.Losses (LossFn)

import Synapse.LinearAlgebra.Mat (Mat)

import Synapse.Autograd (Symbol(unSymbol), constSymbol)


{- | @MetricFn@ type alias represents functions that are able to provide a reference of performance of neural network model.

Every metrix function is expected to return symbol of singleton matrix.
This requirement is not obligatory - but @Synapse@ internally uses this property in @fit@ function.
If you want to bypass this requirement - customise @fit@ function accordingly.
-}
type MetricFn a = Mat a -> Mat a -> Mat a

-- | Converts any loss function to a metric function (because the same constraint is imposed on both @MetricFn@ and @LossFn@).
lossToMetric :: LossFn a -> MetricFn a
lossToMetric loss true predicted = unSymbol $ loss (constSymbol true) (constSymbol predicted)
