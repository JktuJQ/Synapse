{- | This module provides functions that implement neural networks training.
-}


{-# LANGUAGE OverloadedStrings #-}


module Synapse.NN.Training
    ( -- * Training
      Hyperparameters (Hyperparameters, hyperparametersEpochs, hyperparametersBatchSize, hyperparametersLearningRate, hyperparametersLoss, hyperparametersMetrics)
    , RecordedMetrics (RecordedMetrics, unRecordedMetrics)
    , train
    ) where


import Synapse.Tensors.Vec (Vec(Vec), unVec)
import Synapse.Tensors (SingletonOps(unSingleton))

import Synapse.Autograd (Symbolic, unSymbol, constSymbol, getGradientsOf, wrt, symbol)

import Synapse.NN.Layers.Layer (AbstractLayer(..))
import Synapse.NN.Optimizers (Optimizer(..))
import Synapse.NN.Batching (Sample(Sample), unDataset, shuffleDataset, VecDataset, batchVectors)
import Synapse.NN.LearningRates (LearningRate(LearningRate))
import Synapse.NN.Losses (Loss(Loss))
import Synapse.NN.Metrics (Metric(Metric))

import Control.Monad (forM_)
import Data.IORef (newIORef, writeIORef, modifyIORef, readIORef)

import System.Random (RandomGen)

import System.ProgressBar

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
    :: (Symbolic a, Floating a, Ord a, RandomGen g, AbstractLayer model, Optimizer optimizer, Show a)
    => model a                                -- ^ Trained model.
    -> optimizer a                            -- ^ Optimizer that will be used in training.
    -> Hyperparameters a                      -- ^ Hyperparameters of training.
    -> (VecDataset a, g)                      -- ^ Dataset with samples of vector functions, generator of random values that will be used to shuffle dataset.
    -> IO (model a, Vec (RecordedMetrics a), g)  -- ^ Updated model, vector of recorded metrics (loss is also recorded and is the first in vector), updated generator of random values.
train model optimizer (Hyperparameters epochs batchSize (LearningRate lr) (Loss loss) (Vec metrics)) (dataset, gen0) = do
    let totalIterations = epochs * ((V.length (unVec $ unDataset dataset) + batchSize - 1) `div` batchSize)
    progressBar <- newProgressBar
        (defStyle { stylePrefix = exact <> msg " iterations"
                  , stylePostfix = msg "\nElapsed time: " <> elapsedTime renderDuration
                                <> msg "; Remaining time: " <> remainingTime renderDuration ""
                                <> msg "; Total time: " <> totalTime renderDuration ""
                  }
        ) 10 (Progress 0 totalIterations ())

    modelState <- newIORef model
    optimizerParameters <- readIORef modelState >>= newIORef . fmap (optimizerInitialParameters optimizer . unSymbol) . getParameters "m"

    allMetrics <- V.generateM (1 + V.length metrics) (const $ MV.new totalIterations)

    gen <- newIORef gen0

    forM_ [1 .. epochs] $ \epoch -> do

        currentGen <- readIORef gen
        let (shuffledDataset, gen') = shuffleDataset dataset currentGen
        _ <- writeIORef gen gen'

        forM_ (zip [1 ..] $ V.toList $ unVec $ unDataset $ batchVectors batchSize shuffledDataset) $ \(batchI, Sample batchInput batchOutput) -> do
            incProgress progressBar 1
            
            currentModelState <- readIORef modelState
            currentOptimizerParameters <- readIORef optimizerParameters

            let (prediction, regularizersLoss) = symbolicForward "m" (symbol "input" batchInput) currentModelState

            let currentIteration = epoch * batchI
            let lossValue = loss (constSymbol batchOutput) prediction
            let lrValue = lr currentIteration

            let (parameters', optimizerParameters') = unzip $ trainParameters optimizer (lrValue, getGradientsOf $ lossValue + regularizersLoss) (zip (getParameters "m" currentModelState) currentOptimizerParameters)

            _ <- modifyIORef modelState (`updateParameters` parameters')
            _ <- writeIORef optimizerParameters optimizerParameters'

            _ <- MV.write (V.unsafeIndex allMetrics 0) (currentIteration - 1) (unSingleton lossValue)
            forM_ (zip [1 ..] $ V.toList metrics) $
                \(metricI, Metric metric) -> MV.write (V.unsafeIndex allMetrics metricI) (currentIteration - 1) (unSingleton $ metric batchOutput (unSymbol prediction))

    trainedModel <- readIORef modelState
    recordedMetrics <- V.mapM (fmap (RecordedMetrics . Vec) . V.unsafeFreeze) allMetrics
    gen'' <- readIORef gen

    return (trainedModel, Vec recordedMetrics, gen'')
  where
    trainParameters _ _ [] = []
    trainParameters opt (lrValue, gradients) ((parameter, optimizerParameter):xs) =
        optimizerUpdateStep opt (lrValue, unSymbol $ gradients `wrt` parameter) (unSymbol parameter, optimizerParameter)
        : trainParameters opt (lrValue, gradients) xs