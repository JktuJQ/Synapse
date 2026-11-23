module Train (train) where

import Control.Exception (evaluate)
import Control.Monad (foldM, unless, when)
import Control.Monad.Trans.State.Strict (get, put)
import Data.Functor.Identity (Identity (..))
import GHC.TypeNats (KnownNat)
import System.Random (mkStdGen)
import Text.Printf (printf)

import Data (readDataset)
import Model (Parameters, initialParameters, predict)

import Synapse.Autograd qualified as AG
import Synapse.NN qualified as NN
import Synapse.NN.Weights.Safetensors (writeSafetensors)
import Synapse.Tensor qualified as T

train :: T.Backend -> FilePath -> FilePath -> Int -> IO ()
train backend directory weights epochs = do
    unless (epochs > 0) $ fail "Epoch count must be positive"
    training <- readDataset directory "train"

    let batchSize = 100
        trainingPlan = NN.planBatches batchSize NN.KeepLast training
        step = NN.prepareBatchStep backend trainingPlan $ \p x y _ -> NN.runState (trainingStep x y) p

        update (parameters, totalLoss) (Identity batch) = do
            (loss, updated) <- either (fail . show) pure $ NN.runBatch step parameters batch
            value <- evaluate $ scalar loss
            when (isNaN value || isInfinite value) $ fail "Nonfinite training loss"
            nextLoss <- evaluate $ totalLoss + batchWeight batch * value
            pure (updated, nextLoss)

        epoch (parameters, generator) number = do
            let (shuffled, nextGenerator) = NN.runState (NN.shuffle training) generator
            (updated, loss) <- foldM update (parameters, 0) $ NN.loadBatches batchSize NN.KeepLast shuffled
            printf "Epoch %2d | train loss %.4f\n" number (loss / fromIntegral (NN.datasetSize training))
            pure (updated, nextGenerator)

    (parameters, _) <- foldM epoch (T.run backend initialParameters, mkStdGen 42) [1 .. epochs]
    either (fail . show) pure =<< writeSafetensors weights (NN.extractWeights @(Parameters T.Tensor) parameters)
  where
    scalar :: T.TensorData '[] Double -> Double
    scalar = sum . T.toList
    batchWeight batch = NN.withBatch batch $ fromIntegral . NN.validCount

trainingStep
    :: KnownNat batch
    => T.Tensor '[batch, 1, 28, 28] Double
    -> T.Tensor '[batch] Int
    -> NN.State (Parameters T.Tensor) (T.Tensor '[] Double)
trainingStep features targets = do
    parameters <- get
    let objective
            :: AG.Context s
            => Parameters (AG.Tensor s)
            -> AG.Tensor s '[] Double
        objective p = NN.unsafeCrossEntropy @1 T.mean (predict p $ AG.constant features) targets
        (loss, gradients) = AG.backprop objective parameters
    put $ NN.evalState (NN.optimizerStep (NN.SGD 0.1) parameters gradients) ()
    pure loss
