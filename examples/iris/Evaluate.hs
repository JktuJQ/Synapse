module Evaluate (evaluateModel) where

import Text.Printf (printf)

import Data (readDataset)
import Model (Parameters, emptyParameters, predict)
import Synapse.NN qualified as NN
import Synapse.NN.Weights.Safetensors (readSafetensors)
import Synapse.Tensor qualified as T

evaluateModel :: T.Backend -> FilePath -> FilePath -> IO ()
evaluateModel backend dataset path = do
    (_, (features, labels)) <- readDataset dataset
    weights <- either (fail . show) pure =<< readSafetensors path
    parameters <-
        either (fail . show) pure $
            NN.loadWeights @(Parameters T.Tensor) weights (T.run T.CPU emptyParameters)
    let evaluateBatch = T.run3 backend $ \p x y ->
            let logits = predict p x
             in (NN.unsafeCrossEntropy @1 T.mean logits y, NN.accuracy @1 logits y)
        (loss, accuracy) = evaluateBatch parameters features labels
    printf "Test loss %.4f | test accuracy %.1f%%\n" (sum $ T.toList loss) (100 * sum (T.toList accuracy))
