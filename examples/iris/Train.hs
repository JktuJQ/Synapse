module Train (train) where

import Control.Monad (foldM, unless)
import Text.Printf (printf)

import Data (readDataset)
import Model (Parameters, initialParameters, predict)
import Synapse.Autograd qualified as AG
import Synapse.NN qualified as NN
import Synapse.NN.Weights.Safetensors (writeSafetensors)
import Synapse.Tensor qualified as T

trainingStep
    :: Parameters T.Tensor
    -> T.Tensor '[120, 4] Double
    -> T.Tensor '[120] Int
    -> (T.Tensor '[] Double, Parameters T.Tensor)
trainingStep parameters features targets =
    let objective
            :: AG.Context s
            => Parameters (AG.Tensor s)
            -> AG.Tensor s '[] Double
        objective p = NN.unsafeCrossEntropy @1 T.mean (predict p $ AG.constant features) targets
        (loss, gradients) = AG.backprop objective parameters
        updated = NN.evalState (NN.optimizerStep (NN.SGD 0.1) parameters gradients) ()
     in (loss, updated)

train :: T.Backend -> FilePath -> FilePath -> Int -> IO ()
train backend dataset weights epochs = do
    unless (epochs > 0) $ fail "Epoch count must be positive"
    ((features, labels), _) <- readDataset dataset
    let step = T.run backend trainingStep
        epoch parameters number = do
            let (loss, updated) = step parameters features labels
            printf "Epoch %3d | train loss %.4f\n" number (sum $ T.toList loss)
            pure updated
    parameters <- foldM epoch (T.run backend initialParameters) [1 .. epochs]
    either (fail . show) pure =<< writeSafetensors weights (NN.extractWeights @(Parameters T.Tensor) parameters)
