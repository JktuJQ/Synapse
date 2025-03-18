{- | This module provides datatypes and functions that implement neural networks training.
-}


-- | 'OverloadedStrings' are needed to use strings in progress bar.
{-# LANGUAGE OverloadedStrings #-}


module Synapse.NN.Training
    ( -- * 'Callbacks' datatype and associated type aliases

      CallbackFnOnTrainBegin
    , CallbackFnOnEpochBegin
    , CallbackFnOnBatchBegin
    , CallbackFnOnBatchEnd
    , CallbackFnOnEpochEnd
    , CallbackFnOnTrainEnd

    , Callbacks
          ( Callbacks
          , callbacksOnTrainBegin
          , callbacksOnEpochBegin
          , callbacksOnBatchBegin
          , callbacksOnBatchEnd
          , callbacksOnEpochEnd
          , callbacksOnTrainEnd
          )
    , emptyCallbacks

    -- * 'Hyperparameters' datatype

    , Hyperparameters
          ( Hyperparameters
          , hyperparametersEpochs
          , hyperparametersBatchSize
          , hyperparametersDataset
          , hyperparametersLearningRate
          , hyperparametersLoss
          , hyperparametersMetrics
          )

    , RecordedMetric (RecordedMetric, unRecordedMetric)

      -- * Training

    , train
    ) where


import Synapse.Tensors (SingletonOps(unSingleton))
import Synapse.Tensors.Vec (Vec(Vec), unVec)
import Synapse.Tensors.Mat (Mat)

import Synapse.Autograd (Symbolic, unSymbol, constSymbol, getGradientsOf, symbol)

import Synapse.NN.Layers.Layer (AbstractLayer(..))
import Synapse.NN.Optimizers (Optimizer(..), optimizerUpdateParameters)
import Synapse.NN.Batching (Sample(Sample), unDataset, shuffleDataset, VecDataset, batchVectors, BatchedDataset)
import Synapse.NN.LearningRates (LearningRate(LearningRate))
import Synapse.NN.Losses (Loss(Loss))
import Synapse.NN.Metrics (Metric(Metric))

import Data.Functor ((<&>))
import Control.Monad (forM_, when)
import Data.IORef (IORef, newIORef, writeIORef, readIORef, modifyIORef')

import System.Random (RandomGen)

import System.ProgressBar

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV


-- | Type of callback that is called at the beginning of training.
type CallbackFnOnTrainBegin model optimizer a
    =  IORef (model a)                          -- ^ Initial model state.
    -> IORef [OptimizerParameters optimizer a]  -- ^ Initial optimizer parameters.
    -> IO ()

-- | Type of callback that is called at the beginning of training epoch.
type CallbackFnOnEpochBegin model optimizer a
    =  IORef Int                                -- ^ Current epoch.
    -> IORef (model a)                          -- ^ Model state at the beginning of the epoch processing.
    -> IORef [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the beginning of the epoch processing.
    -> IORef (BatchedDataset a)                 -- ^ Batched shuffled dataset.
    -> IO ()

-- | Type of callback that is called at the beginning of training batch.
type CallbackFnOnBatchBegin model optimizer a
    =  IORef Int                                -- ^ Current epoch.
    -> IORef Int                                -- ^ Current batch number.
    -> IORef (model a)                          -- ^ Model state at the beginning of the batch processing.
    -> IORef [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the beginning of the batch processing.
    -> IORef (Sample (Mat a))                   -- ^ Batch that is being processed.
    -> IORef a                                  -- ^ Learning rate value.
    -> IO ()

-- | Type of callback that is called at the end of training batch.
type CallbackFnOnBatchEnd model optimizer a
    =  IORef Int                                -- ^ Current epoch.
    -> IORef Int                                -- ^ Current batch number.
    -> IORef (model a)                          -- ^ Model state at the end of the batch processing.
    -> IORef [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the end of the batch processing.
    -> IORef (Vec a)                            -- ^ Metrics that were recorded on this batch.
    -> IO ()

-- | Type of callback that is called at the end of training epoch.
type CallbackFnOnEpochEnd model optimizer a
    =  IORef Int                                -- ^ Current epoch.
    -> IORef (model a)                          -- ^ Model state at the end of the epoch processing.
    -> IORef [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the end of the epoch processing.
    -> IO ()

-- | Type of callback that is called at the end of training.
type CallbackFnOnTrainEnd model optimizer a
    =  IORef (model a)                            -- ^ Model state at the end of the training.
    -> IORef [OptimizerParameters optimizer a]    -- ^ Optimizer parameters at the end of the training.
    -> IORef (Vec (RecordedMetric a))             -- ^ Recorded metrics.
    -> IO ()

-- | 'Callbacks' record datatype holds all callbacks for the training.
data Callbacks model optimizer a = Callbacks
    { callbacksOnTrainBegin :: [CallbackFnOnTrainBegin model optimizer a]  -- ^ Callbacks that will be called at the beginning of training.
    , callbacksOnEpochBegin :: [CallbackFnOnEpochBegin model optimizer a]  -- ^ Callbacks that will be called at the beginning of training epoch processing.
    , callbacksOnBatchBegin :: [CallbackFnOnBatchBegin model optimizer a]  -- ^ Callbacks that will be called at the beginning of training batch processing.
    , callbacksOnBatchEnd   :: [CallbackFnOnBatchEnd   model optimizer a]  -- ^ Callbacks that will be called at the end of training batch processing.
    , callbacksOnEpochEnd   :: [CallbackFnOnEpochEnd   model optimizer a]  -- ^ Callbacks that will be called at the end of training epoch processing.
    , callbacksOnTrainEnd   :: [CallbackFnOnTrainEnd   model optimizer a]  -- ^ Callbacks that will be called at the end of training.
    }

-- | Returns empty 'Callbacks' record. It could also be used to build your own callbacks upon.
emptyCallbacks :: Callbacks model optimizer a
emptyCallbacks = Callbacks [] [] [] [] [] []


-- | 'Hyperparameters' datatype represents configuration of a training.
data Hyperparameters a = Hyperparameters
    { hyperparametersEpochs       :: Int             -- ^ Number of epochs in the training.
    , hyperparametersBatchSize    :: Int             -- ^ Size of batches that will be used in the training.

    , hyperparametersDataset      :: VecDataset a    -- ^ Dataset with samples of vector functions.

    , hyperparametersLearningRate :: LearningRate a  -- ^ 'LearningRate' that will be used in the training.
    , hyperparametersLoss         :: Loss a          -- ^ 'Loss' that will be used in the training.

    , hyperparametersMetrics      :: Vec (Metric a)  -- ^ 'Metric's that will be recorded during training.
    }

-- | 'RecordedMetric' newtype wraps vector of results of metrics.
newtype RecordedMetric a = RecordedMetric
    { unRecordedMetric :: Vec a  -- ^ Results of metric recording.
    }


-- | 'whileM_' function implements a monadic @while@ loop which can be @break@ed if the condition becomes false.
whileM_ :: (Monad m) => m Bool -> m a -> m ()
whileM_ p f = go
  where
    go = p >>= flip when (f >> go)

-- | 'train' function is the heart of "Synapse" library. It allows training neural networks on datasets with specified parameters.
train
    :: (Symbolic a, Floating a, Ord a, Show a, RandomGen g, AbstractLayer model, Optimizer optimizer)
    => model a                                                                     -- ^ Trained model.
    -> optimizer a                                                                 -- ^ Optimizer that will be during training.
    -> Hyperparameters a                                                           -- ^ Hyperparameters of training.
    -> Callbacks model optimizer a                                                 -- ^ Callbacks that will be used during training.
    -> g                                                                           -- ^ Generator of random values that will be used to shuffle dataset.
    -> IO (model a, [OptimizerParameters optimizer a], Vec (RecordedMetric a), g)  -- ^ Updated model, optimizer parameters at the end of training, vector of recorded metrics (loss is also recorded and is the first in vector), updated generator of random values.
train model optimizer (Hyperparameters epochs batchSize dataset (LearningRate lr) (Loss loss) (Vec metrics)) callbacks gen0 = do
    let totalIterations = epochs * ((V.length (unVec $ unDataset dataset) + batchSize - 1) `div` batchSize)
    progressBar <- newProgressBar
        (defStyle { stylePrefix = exact <> msg " iterations"
                  , stylePostfix = msg "\nElapsed time: " <> elapsedTime renderDuration
                                <> msg "; Remaining time: " <> remainingTime renderDuration ""
                                <> msg "; Total time: " <> totalTime renderDuration ""
                  }
        ) 10 (Progress 0 totalIterations ())

    modelRef <- newIORef model

    optimizerParametersRef <- newIORef $ optimizerInitialParameters optimizer . unSymbol <$> getParameters "m" model

    mapM_ (\fn -> fn modelRef optimizerParametersRef) (callbacksOnTrainBegin callbacks)

    allMetrics <- V.generateM (1 + V.length metrics) (const $ MV.new totalIterations)

    gen <- newIORef gen0


    epochRef <- newIORef 1
    whileM_ (readIORef epochRef <&> (<= epochs)) $ do

        currentGen <- readIORef gen
        let (shuffledDataset, gen') = shuffleDataset dataset currentGen
        _ <- writeIORef gen gen'

        batchedDatasetRef <- newIORef $ batchVectors batchSize shuffledDataset

        mapM_ (\fn -> fn epochRef modelRef optimizerParametersRef batchedDatasetRef) (callbacksOnEpochBegin callbacks)

        epoch <- readIORef epochRef

        batchedDataset <- readIORef batchedDatasetRef
        batchIRef <- newIORef 1
        whileM_ (readIORef batchIRef <&> (<= V.length (unVec $ unDataset batchedDataset))) $ do
            incProgress progressBar 1

            batchI <- readIORef batchIRef
            let currentIteration = epoch * batchI

            batchRef <- newIORef $ V.unsafeIndex (unVec $ unDataset batchedDataset) (batchI - 1)
            lrValueRef <- newIORef $ lr currentIteration

            mapM_ (\fn -> fn epochRef batchIRef modelRef optimizerParametersRef batchRef lrValueRef) (callbacksOnBatchBegin callbacks)

            (Sample batchInput batchOutput) <- readIORef batchRef

            (prediction, regularizersLoss) <- readIORef modelRef <&> symbolicForward "m" (symbol "input" batchInput)

            lrValue <- readIORef lrValueRef
            let lossValue = loss (constSymbol batchOutput) prediction

            parameters <- readIORef modelRef <&> getParameters "m"
            optimizerParameters <- readIORef optimizerParametersRef
            let (parameters', optimizerParameters') = unzip $ optimizerUpdateParameters optimizer (lrValue, getGradientsOf $ lossValue + regularizersLoss)
                                                                                        (zip parameters optimizerParameters)

            _ <- modifyIORef' modelRef (`updateParameters` parameters')
            _ <- writeIORef optimizerParametersRef optimizerParameters'

            metricsValuesRef <- newIORef $ Vec $ V.cons (unSingleton lossValue) $ V.map (\(Metric metric) -> unSingleton $ metric batchOutput (unSymbol prediction)) metrics

            mapM_ (\fn -> fn epochRef batchIRef modelRef optimizerParametersRef metricsValuesRef) (callbacksOnBatchEnd callbacks)

            (Vec metricsValues) <- readIORef metricsValuesRef
            forM_ [0 .. (V.length metricsValues - 1)] $ \metricI -> MV.write (V.unsafeIndex allMetrics metricI) (currentIteration - 1) (V.unsafeIndex metricsValues metricI)

            modifyIORef' batchIRef (+ 1)

        mapM_ (\fn -> fn epochRef modelRef optimizerParametersRef) (callbacksOnEpochEnd callbacks)

        modifyIORef' epochRef (+ 1)


    recordedMetricsRef <- V.mapM (fmap (RecordedMetric . Vec) . V.unsafeFreeze) allMetrics >>= newIORef . Vec

    mapM_ (\fn -> fn modelRef optimizerParametersRef recordedMetricsRef) (callbacksOnTrainEnd callbacks)

    trainedModel <- readIORef modelRef
    trainedOptimizerParameters <- readIORef optimizerParametersRef
    recordedMetrics <- readIORef recordedMetricsRef
    gen'' <- readIORef gen

    return (trainedModel, trainedOptimizerParameters, recordedMetrics, gen'')    
