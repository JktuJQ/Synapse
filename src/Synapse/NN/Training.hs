{- | This module provides functions that implement neural networks training.
-}


{-# LANGUAGE OverloadedStrings #-}


module Synapse.NN.Training
    ( -- * @Callbacks@ datatype and associated type aliases

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

    -- * @Hyperparameters@ datatype

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

import Synapse.Autograd (Symbolic, unSymbol, constSymbol, getGradientsOf, wrt, symbol, SymbolMat)

import Synapse.NN.Layers.Layer (AbstractLayer(..))
import Synapse.NN.Optimizers (Optimizer(..))
import Synapse.NN.Batching (Sample(Sample), unDataset, shuffleDataset, VecDataset, batchVectors, BatchedDataset)
import Synapse.NN.LearningRates (LearningRate(LearningRate))
import Synapse.NN.Losses (Loss(Loss))
import Synapse.NN.Metrics (Metric(Metric))

import Data.Functor ((<&>))
import Control.Monad (forM_)
import Data.IORef (newIORef, writeIORef, readIORef)

import System.Random (RandomGen)

import System.ProgressBar

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV


-- | Type of callback that is called at the beginning of training.
type CallbackFnOnTrainBegin model optimizer a
    =  model a                            -- ^ Initial model state.
    -> [OptimizerParameters optimizer a]  -- ^ Initial optimizer parameters.
    -> IO ()

-- | Type of callback that is called at the beginning of training epoch.
type CallbackFnOnEpochBegin model optimizer a
    =  Int                                -- ^ Current epoch.
    -> model a                            -- ^ Model state at the beginning of the epoch processing.
    -> [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the beginning of the epoch processing.
    -> BatchedDataset a                   -- ^ Batched shuffled dataset.
    -> IO ()

-- | Type of callback that is called at the beginning of training batch.
type CallbackFnOnBatchBegin model optimizer a
    =  Int                                -- ^ Current epoch.
    -> Int                                -- ^ Current batch number.
    -> model a                            -- ^ Model state at the beginning of the batch processing.
    -> [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the beginning of the batch processing.
    -> Sample (Mat a)                     -- ^ Batch that is being processed.
    -> (SymbolMat a, SymbolMat a)         -- ^ Prediction and regularizer output.
    -> a                                  -- ^ Learning rate value.
    -> SymbolMat a                        -- ^ Loss value.
    -> Vec a                              -- ^ Metrics on this batch.
    -> IO ()

-- | Type of callback that is called at the end of training batch.
type CallbackFnOnBatchEnd model optimizer a
    =  Int                                -- ^ Current epoch.
    -> Int                                -- ^ Current batch number.
    -> model a                            -- ^ Model state at the end of the batch processing.
    -> [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the end of the batch processing.
    -> IO ()

-- | Type of callback that is called at the end of training epoch.
type CallbackFnOnEpochEnd model optimizer a
    =  Int                                -- ^ Current epoch.
    -> model a                            -- ^ Model state at the end of the epoch processing.
    -> [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the end of the epoch processing.
    -> IO ()

-- | Type of callback that is called at the end of training.
type CallbackFnOnTrainEnd model optimizer a
    =  model a                            -- ^ Model state at the end of the training.
    -> [OptimizerParameters optimizer a]  -- ^ Optimizer parameters at the end of the training.
    -> Vec (RecordedMetric a)             -- ^ Recorded metrics.
    -> IO ()

-- | @Callbacks@ record datatype holds all callbacks for the training.
data Callbacks model optimizer a = Callbacks
    { callbacksOnTrainBegin :: [CallbackFnOnTrainBegin model optimizer a]  -- ^ Callbacks that will be called at the beginning of training.
    , callbacksOnEpochBegin :: [CallbackFnOnEpochBegin model optimizer a]  -- ^ Callbacks that will be called at the beginning of training epoch processing.
    , callbacksOnBatchBegin :: [CallbackFnOnBatchBegin model optimizer a]  -- ^ Callbacks that will be called at the beginning of training batch processing.
    , callbacksOnBatchEnd   :: [CallbackFnOnBatchEnd   model optimizer a]  -- ^ Callbacks that will be called at the end of training batch processing.
    , callbacksOnEpochEnd   :: [CallbackFnOnEpochEnd   model optimizer a]  -- ^ Callbacks that will be called at the end of training epoch processing.
    , callbacksOnTrainEnd   :: [CallbackFnOnTrainEnd   model optimizer a]  -- ^ Callbacks that will be called at the end of training.
    }

-- | Returns empty @Callbacks@ record. It could also be used to build your own callbacks upon.
emptyCallbacks :: Callbacks model optimizer a
emptyCallbacks = Callbacks [] [] [] [] [] []


-- | @Hyperparamters@ datatype represents configuration of a training.
data Hyperparameters a = Hyperparameters
    { hyperparametersEpochs       :: Int             -- ^ Number of epochs in the training.
    , hyperparametersBatchSize    :: Int             -- ^ Size of batches that will be used in the training.

    , hyperparametersDataset      :: VecDataset a    -- ^ Dataset with samples of vector functions.

    , hyperparametersLearningRate :: LearningRate a  -- ^ @LearningRate@ that will be used in the training.
    , hyperparametersLoss         :: Loss a          -- ^ @Loss@ that will be used in the training.

    , hyperparametersMetrics      :: Vec (Metric a)  -- ^ @Metric@s that will be recorded during training.
    }

-- | @RecordedMetrics@ newtype wraps vector of results of metrics.
newtype RecordedMetric a = RecordedMetric
    { unRecordedMetric :: Vec a  -- ^ Results of metric recording.
    }


-- | @train@ function is the heart of @Synapse@ library. It allows training neural networks on datasets.
train
    :: (Symbolic a, Floating a, Ord a, Show a, RandomGen g, AbstractLayer model, Optimizer optimizer)
    => model a                                  -- ^ Trained model.
    -> optimizer a                              -- ^ Optimizer that will be during training.
    -> Hyperparameters a                        -- ^ Hyperparameters of training.
    -> Callbacks model optimizer a              -- ^ Callbacks that will be used during training.
    -> g                                        -- ^ Generator of random values that will be used to shuffle dataset.
    -> IO (model a, Vec (RecordedMetric a), g)  -- ^ Updated model, vector of recorded metrics (loss is also recorded and is the first in vector), updated generator of random values.
train model optimizer (Hyperparameters epochs batchSize dataset (LearningRate lr) (Loss loss) (Vec metrics)) callbacks gen0 = do
    let totalIterations = epochs * ((V.length (unVec $ unDataset dataset) + batchSize - 1) `div` batchSize)
    progressBar <- newProgressBar
        (defStyle { stylePrefix = exact <> msg " iterations"
                  , stylePostfix = msg "\nElapsed time: " <> elapsedTime renderDuration
                                <> msg "; Remaining time: " <> remainingTime renderDuration ""
                                <> msg "; Total time: " <> totalTime renderDuration ""
                  }
        ) 10 (Progress 0 totalIterations ())

    modelState <- newIORef model

    let initialOptimizerParameters = optimizerInitialParameters optimizer . unSymbol <$> getParameters "m" model
    optimizerParameters <- newIORef initialOptimizerParameters

    mapM_ (\fn -> fn model initialOptimizerParameters) (callbacksOnTrainBegin callbacks)

    allMetrics <- V.generateM (1 + V.length metrics) (const $ MV.new totalIterations)

    gen <- newIORef gen0

    forM_ [1 .. epochs] $ \epoch -> do

        currentGen <- readIORef gen
        let (shuffledDataset, gen') = shuffleDataset dataset currentGen
        _ <- writeIORef gen gen'

        let batchedDataset = batchVectors batchSize shuffledDataset

        epochBeginningModelState <- readIORef modelState
        epochBeginningOptimizerParameters <- readIORef optimizerParameters

        mapM_ (\fn -> fn epoch epochBeginningModelState epochBeginningOptimizerParameters batchedDataset) (callbacksOnEpochBegin callbacks)

        forM_ (zip [1 ..] $ V.toList $ unVec $ unDataset batchedDataset) $ \(batchI, batch@(Sample batchInput batchOutput)) -> do
            incProgress progressBar 1

            batchBeginningModelState <- readIORef modelState
            batchBeginningOptimizerParameters <- readIORef optimizerParameters

            let (prediction, regularizersLoss) = symbolicForward "m" (symbol "input" batchInput) batchBeginningModelState

            let currentIteration = epoch * batchI
            let lrValue = lr currentIteration

            let lossValue = loss (constSymbol batchOutput) prediction
            _ <- MV.write (V.unsafeIndex allMetrics 0) (currentIteration - 1) (unSingleton lossValue)

            let currentMetrics = V.map (\(Metric metric) -> unSingleton $ metric batchOutput (unSymbol prediction)) metrics

            forM_ (zip [1 ..] $ V.toList currentMetrics) $
                \(metricI, metricValue) -> MV.write (V.unsafeIndex allMetrics metricI) (currentIteration - 1) metricValue

            mapM_ (\fn -> fn epoch batchI batchBeginningModelState batchBeginningOptimizerParameters batch (prediction, regularizersLoss) lrValue lossValue (Vec $ unSingleton lossValue `V.cons` currentMetrics)) (callbacksOnBatchBegin callbacks)

            let (parameters', optimizerParameters') = unzip $ trainParameters optimizer (lrValue, getGradientsOf $ lossValue + regularizersLoss) (zip (getParameters "m" batchBeginningModelState) batchBeginningOptimizerParameters)

            let updatedModelState = batchBeginningModelState `updateParameters` parameters'
            _ <- writeIORef modelState updatedModelState
            _ <- writeIORef optimizerParameters optimizerParameters'

            mapM_ (\fn -> fn epoch batchI updatedModelState optimizerParameters') (callbacksOnBatchEnd callbacks)

        epochEndModelState <- readIORef modelState
        epochEndOptimizerParameters <- readIORef optimizerParameters

        mapM_ (\fn -> fn epoch epochEndModelState epochEndOptimizerParameters) (callbacksOnEpochEnd callbacks)


    trainedModel <- readIORef modelState
    trainEndOptimizerParameters <- readIORef optimizerParameters
    recordedMetrics <- V.mapM (fmap (RecordedMetric . Vec) . V.unsafeFreeze) allMetrics <&> Vec
    gen'' <- readIORef gen

    mapM_ (\fn -> fn trainedModel trainEndOptimizerParameters recordedMetrics) (callbacksOnTrainEnd callbacks)

    return (trainedModel, recordedMetrics, gen'')
  where
    trainParameters _ _ [] = []
    trainParameters opt (lrValue, gradients) ((parameter, optimizerParameter):xs) =
        optimizerUpdateStep opt (lrValue, unSymbol $ gradients `wrt` parameter) (unSymbol parameter, optimizerParameter)
        : trainParameters opt (lrValue, gradients) xs
