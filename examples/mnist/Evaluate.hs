module Evaluate (evaluateModel) where

import Control.Monad (foldM)
import Data.Functor.Identity (Identity (..))
import Text.Printf (printf)

import Data (readDataset)
import Model (Parameters, emptyParameters, predict)
import Synapse.NN qualified as NN
import Synapse.NN.Weights.Safetensors (readSafetensors)
import Synapse.Tensor qualified as T

evaluateModel :: T.Backend -> FilePath -> FilePath -> IO ()
evaluateModel backend dataset path = do
    testing <- readDataset dataset "t10k"
    weights <- either (fail . show) pure =<< readSafetensors path
    parameters <-
        either (fail . show) pure $
            NN.loadWeights @(Parameters T.Tensor) weights (T.run T.defaultCPU emptyParameters)
    let plan = NN.planBatches 100 NN.KeepLast testing
        measure = NN.prepareBatchStep backend plan $ \p x y _ ->
            let logits = predict p x
             in (NN.unsafeCrossEntropy @1 T.mean logits y, NN.accuracy @1 logits y)
        accumulate (!totalLoss, !totalAccuracy) (Identity batch) = do
            (loss, accuracy) <- either (fail . show) pure $ NN.runBatch measure parameters batch
            let weight = NN.withBatch batch $ fromIntegral . NN.validCount
            pure (totalLoss + weight * sum (T.toList loss), totalAccuracy + weight * sum (T.toList accuracy))
        count = fromIntegral $ NN.datasetSize testing
    (loss, accuracy) <- foldM accumulate (0, 0) $ NN.batchLoads plan
    printf "Test loss %.4f | test accuracy %.1f%%\n" (loss / count) (100 * accuracy / count)
