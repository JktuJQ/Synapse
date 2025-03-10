{- | This module provides functions that implement neural networks training.
-}


module Synapse.NN.Training
    ( -- * Training
      Hyperparameters (Hyperparameters, hyperparametersEpochs, hyperparametersBatchSize, hyperparametersLearningRate, hyperparametersLoss, hyperparametersMetrics)
    , RecordedMetrics (RecordedMetrics, unRecordedMetrics)
    , fit
    ) where


import Synapse.Tensors (SingletonOps(unSingleton))

import Synapse.Autograd (Symbolic, unSymbol, constSymbol, symbol, getGradientsOf, wrt)

import Synapse.NN.Layers.Layer (AbstractLayerM(..))
import Synapse.NN.Models (Model(Training))
import Synapse.NN.Optimizers (Optimizer(..))
import Synapse.NN.Batching (Sample(Sample), Dataset(unDataset), shuffleDataset, VecDataset, batchVectors)
import Synapse.NN.LearningRates (LearningRate(LearningRate))
import Synapse.NN.Losses (Loss(Loss))
import Synapse.NN.Metrics (Metric(Metric))

import Control.Monad (forM, forM_)

import System.Random (RandomGen)

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV


-- | @MRef@ is an analog of @IORef@/@STRef@ for any monadic context. It is implemented as a one-element mutable vector.
newtype MRef m a = MRef
    { unMRef :: MV.MVector (MV.PrimState m) a  -- ^ Unwraps @MRef@ newtype.
    }

-- | Creates new @MRef@.
mrefNew :: MV.PrimMonad m => a -> m (MRef m a)
mrefNew x = do
    mref <- MV.new 1
    _ <- MV.write mref 0 x
    return $ MRef mref

-- | Reads from @MRef@.
mrefRead :: MV.PrimMonad m => MRef m a -> m a
mrefRead = flip MV.read 0 . unMRef

-- | Write to @MRef@.
mrefWrite :: MV.PrimMonad m => MRef m a -> a -> m ()
mrefWrite = flip MV.write 0 . unMRef


-- | @Hyperparamters@ datatype represents configuration of a training.
data Hyperparameters a = Hyperparameters
    { hyperparametersEpochs       :: Int                  -- ^ Number of epochs in the training.
    , hyperparametersBatchSize    :: Int                  -- ^ Size of batches that will be used in the training.

    , hyperparametersLearningRate :: LearningRate a       -- ^ @LearningRate@ that will be used in the training.
    , hyperparametersLoss         :: Loss a               -- ^ @Loss@ that will be used in the training.
    , hyperparametersMetrics      :: V.Vector (Metric a)  -- ^ @Metric@s that will be recorded during training.
    }

-- | @RecordedMetrics@ newtype wraps vector of results of metrics.
newtype RecordedMetrics a = RecordedMetrics
    { unRecordedMetrics :: V.Vector a  -- ^ Results of metric recording.
    }

-- | @fit@ function is the heart of @Synapse@ library. It allows training neural networks on datasets.
fit 
    :: (Symbolic a, Floating a, Ord a, MV.PrimMonad m, RandomGen g, Model model m, Optimizer optimizer)
    => Training model m a                                       -- ^ Trained model.
    -> optimizer a                                              -- ^ Optimizer that will be used in training.
    -> Hyperparameters a                                        -- ^ Hyperparameters of training.
    -> (VecDataset a, g)                                        -- ^ Dataset with samples of vector functions, generator of random values that will be used to shuffle dataset.
    -> m (Training model m a, V.Vector (RecordedMetrics a), g)  -- ^ Updated model, vector of recorded metrics (loss is also recorded and is the first in vector), updated generator of random values.
fit model optimizer (Hyperparameters epochs batchSize (LearningRate lr) (Loss loss) metrics) (dataset, gen0) =
    let modelPrefix = "m"
        inputPrefix = "input"
    in do
    modelParameters <- getParametersM modelPrefix model
    optimizerParameters <- V.thaw $ V.fromList $ fmap (optimizerInitialParameters optimizer . unSymbol) modelParameters

    allMetrics <- V.generateM (1 + V.length metrics) (const $ MV.new (epochs * (V.length (unDataset dataset) `div` batchSize)))

    gen <- mrefNew gen0
    iteration <- mrefNew (0 :: Int)

    forM_ [1 .. epochs] $ \_ -> do
        currentGen <- mrefRead gen
        let (shuffledDataset, gen') = shuffleDataset dataset currentGen
        _ <- mrefWrite gen gen'

        let batches = batchVectors batchSize shuffledDataset
        forM_ (V.toList $ unDataset batches) $ \(Sample batchInput batchOutput) -> do
            prediction <- symbolicForwardM modelPrefix (symbol inputPrefix batchInput) model
            regularizerLoss <- applyRegularizerM modelPrefix model

            let lossValue = loss (constSymbol batchOutput) prediction

            currentIteration <- mrefRead iteration
            _ <- mrefWrite iteration (currentIteration + 1)

            MV.write (V.unsafeIndex allMetrics 0) currentIteration (unSingleton lossValue)
            forM_ (zip [1 :: Int ..] $ V.toList metrics) $
                \(i, Metric metric) -> MV.write (V.unsafeIndex allMetrics i) currentIteration (unSingleton $ metric batchOutput (unSymbol prediction))

            let lrValue = lr currentIteration
            let lossWithRegularizers = lossValue + regularizerLoss
            let gradients = getGradientsOf lossWithRegularizers

            modelParameters' <- forM (zip [0 :: Int ..] modelParameters) $ \(i, parameter) -> do
                currentOptimizerParameter <- MV.read optimizerParameters i
                let (parameter', optimizerParameter') =
                     optimizerUpdateStep optimizer (unSymbol parameter, currentOptimizerParameter) (lrValue, unSymbol $ gradients `wrt` parameter)
                _ <- MV.write optimizerParameters i optimizerParameter'
                return parameter'

            updateParametersM model modelParameters'


    recordedMetrics <- V.mapM (fmap RecordedMetrics . V.unsafeFreeze) allMetrics
    gen'' <- mrefRead gen

    return (model, recordedMetrics, gen'')
