{- | This module provides functions that implement neural networks training.
-}


module Synapse.NN.Training
    ( -- * Training
      Hyperparameters (Hyperparameters, hyperparametersEpochs, hyperparametersBatchSize, hyperparametersLearningRate, hyperparametersLoss, hyperparametersMetrics)
    , RecordedMetrics (RecordedMetrics, unRecordedMetrics)
    , train
    ) where


import Synapse.Tensors.Vec (Vec(Vec), unVec)
import Synapse.Tensors (SingletonOps(unSingleton))

import Synapse.Autograd (Symbolic, unSymbol, constSymbol, getGradientsOf, wrt)

import Synapse.NN.Layers.Layer (AbstractLayer(..))
import Synapse.NN.Optimizers (Optimizer(..))
import Synapse.NN.Batching (Sample(Sample), unDataset, shuffleDataset, VecDataset, batchVectors)
import Synapse.NN.LearningRates (LearningRate(LearningRate))
import Synapse.NN.Losses (Loss(Loss))
import Synapse.NN.Metrics (Metric(Metric))

import Control.Monad (forM_)
import Control.Monad.ST (runST)
import Data.STRef (newSTRef, writeSTRef, modifySTRef, readSTRef)

import System.Random (RandomGen)

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV


-- | @Hyperparamters@ datatype represents configuration of a training.
data Hyperparameters a = Hyperparameters
    { hyperparametersEpochs       :: Int             -- ^ Number of epochs in the training.
    , hyperparametersBatchSize    :: Int             -- ^ Size of batches that will be used in the training.

    , hyperparametersLearningRate :: LearningRate a  -- ^ @LearningRate@ that will be used in the training.
    , hyperparametersLoss         :: Loss a          -- ^ @Loss@ that will be used in the training.
    , hyperparametersMetrics      :: Vec (Metric a)  -- ^ @Metric@s that will be recorded during training.
    }

-- | @RecordedMetrics@ newtype wraps vector of results of metrics.
newtype RecordedMetrics a = RecordedMetrics
    { unRecordedMetrics :: Vec a  -- ^ Results of metric recording.
    }


-- | @train@ function is the heart of @Synapse@ library. It allows training neural networks on datasets.
train
    :: (Symbolic a, Floating a, Ord a, RandomGen g, AbstractLayer model, Optimizer optimizer)
    => model a                                -- ^ Trained model.
    -> optimizer a                            -- ^ Optimizer that will be used in training.
    -> Hyperparameters a                      -- ^ Hyperparameters of training.
    -> (VecDataset a, g)                      -- ^ Dataset with samples of vector functions, generator of random values that will be used to shuffle dataset.
    -> (model a, Vec (RecordedMetrics a), g)  -- ^ Updated model, vector of recorded metrics (loss is also recorded and is the first in vector), updated generator of random values.
train model optimizer (Hyperparameters epochs batchSize (LearningRate lr) (Loss loss) (Vec metrics)) (dataset, gen0) = runST $ do
    modelState <- newSTRef model
    optimizerParameters <- readSTRef modelState >>= newSTRef . fmap (optimizerInitialParameters optimizer . unSymbol) . getParameters "m"

    let batchesN = (V.length (unVec $ unDataset dataset) + batchSize - 1) `div` batchSize

    allMetrics <- V.generateM (1 + V.length metrics) (const $ MV.new (epochs * batchesN))

    gen <- newSTRef gen0

    forM_ [1 .. epochs] $ \epoch -> do

        currentGen <- readSTRef gen
        let (shuffledDataset, gen') = shuffleDataset dataset currentGen
        _ <- writeSTRef gen gen'

        forM_ (zip [1 ..] $ V.toList $ unVec $ unDataset $ batchVectors batchSize shuffledDataset) $ \(batchI, Sample batchInput batchOutput) -> do
            currentModelState <- readSTRef modelState
            currentOptimizerParameters <- readSTRef optimizerParameters

            let (prediction, regularizersLoss) = symbolicForward "m" (constSymbol batchInput) currentModelState

            let currentIteration = epoch * batchI
            let lossValue = loss (constSymbol batchOutput) prediction
            let lrValue = lr currentIteration

            let (parameters', optimizerParameters') = unzip $ trainParameters optimizer (lrValue, getGradientsOf $ lossValue + regularizersLoss) (zip (getParameters "m" currentModelState) currentOptimizerParameters)

            _ <- modifySTRef modelState (`updateParameters` parameters')
            _ <- writeSTRef optimizerParameters optimizerParameters'

            MV.write (V.unsafeIndex allMetrics 0) currentIteration (unSingleton lossValue)
            forM_ (zip [1 ..] $ V.toList metrics) $
                \(metricI, Metric metric) -> MV.write (V.unsafeIndex allMetrics metricI) currentIteration (unSingleton $ metric batchOutput (unSymbol prediction))

    recordedMetrics <- V.mapM (fmap (RecordedMetrics . Vec) . V.unsafeFreeze) allMetrics
    gen'' <- readSTRef gen

    return (model, Vec recordedMetrics, gen'')
  where
    trainParameters _ _ [] = []
    trainParameters opt (lrValue, gradients) ((parameter, optimizerParameter):xs) =
        optimizerUpdateStep opt (lrValue, unSymbol $ gradients `wrt` parameter) (unSymbol parameter, optimizerParameter)
        : trainParameters opt (lrValue, gradients) xs