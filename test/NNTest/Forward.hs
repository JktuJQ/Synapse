{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module NNTest.Forward (tests) where

import Control.Arrow (Arrow, arr, left, (&&&), (>>>))
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Ops.Elementwise as E
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Parameters t = Parameters
  { projection :: NN.Dense 2 2 t Double,
    normalization :: NN.BatchNorm 2 t Double,
    trailing :: NN.LayerNorm '[2] t Double,
    output :: NN.Dense 2 1 t Double
  }
  deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor) = Parameters t

data NetworkState t = NetworkState
  { statistics :: NN.BatchNormState 2 t Double,
    randomness :: NN.DropoutState t,
    marker :: t '[] Double
  }
  deriving (Generic)

type instance T.WithTensor t (NetworkState T.Tensor) = NetworkState t

statisticsLens :: (Functor f) => (NN.BatchNormState 2 t Double -> f (NN.BatchNormState 2 t Double)) -> NetworkState t -> f (NetworkState t)
statisticsLens change whole = fmap (\updated -> whole {statistics = updated}) $ change $ statistics whole

randomnessLens :: (Functor f) => (NN.DropoutState t -> f (NN.DropoutState t)) -> NetworkState t -> f (NetworkState t)
randomnessLens change whole = fmap (\updated -> whole {randomness = updated}) $ change $ randomness whole

firstLens :: (Functor f) => (a -> f a) -> (a, b) -> f (a, b)
firstLens change (a, b) = (,b) <$> change a

predict ::
  ( T.TensorOps t Double,
    Floating (t '[2, 2] Double),
    Num (t '[2, 1] Double),
    E.Conditional (t '[2, 2] Double),
    E.Condition (t '[2, 2] Double) ~ T.Tensor '[2, 2] Bool
  ) =>
  NN.ExecutionMode ->
  Parameters t ->
  NN.Forward (NetworkState T.Tensor) (t '[2, 2] Double) (t '[2, 1] Double)
predict mode parameters = proc input -> do
  projected <- NN.dense (projection parameters) -< input
  normalized <- NN.stateful statisticsLens (NN.batchNorm @1 mode 0.1 0.5 $ normalization parameters) -< projected
  activated <- NN.tanh -< normalized
  rescaled <- NN.layerNorm 0.1 (trailing parameters) -< activated
  masked <- NN.stateful randomnessLens (NN.dropout mode 0.25) -< rescaled
  NN.dense (output parameters) -< masked

stateless :: (Arrow arrow) => NN.Dense 2 1 T.Tensor Double -> arrow (T.Tensor '[2] Double) (T.Tensor '[1] Double)
stateless parameters = proc input -> do
  projected <- NN.dense parameters -< input
  NN.tanh -< projected

initialParameters :: Parameters T.Tensor
initialParameters = Parameters (NN.Dense (T.fromList [1, 0, 0, 1]) 0) (NN.BatchNorm 1 0) (NN.LayerNorm 1 0) (NN.Dense (T.fromList [2, -1]) 0)

initialNetworkState :: NetworkState T.Tensor
initialNetworkState = NetworkState NN.initialBatchNormState (NN.DropoutState $ T.generatorFromSeed 42) 17

tests :: TestTree
tests =
  testGroup
    "forward"
    [ testProperty "ordinary arrows preserve state and compose without adapters" $ H.withTests 1 $ H.property $ do
        let pipeline = arr (+ 1) >>> arr (* 2) :: NN.Forward Int Int Int
        NN.forward pipeline 3 17 H.=== (8, 17),
      testProperty "stateful lenses update only the selected field" $ H.withTests 1 $ H.property $ do
        let operation previous input = (previous + input, previous + 1)
            pipeline = NN.stateful firstLens operation
        NN.forward pipeline (3 :: Int) (2, True) H.=== (5, (3, True)),
      testProperty "branches share updated state in left-to-right order" $ H.withTests 1 $ H.property $ do
        let operation previous input = (previous + input, previous + 1)
            branch = NN.stateful firstLens operation
        NN.forward (branch &&& branch) (10 :: Int) (0, True) H.=== ((10, 11), (2, True)),
      testProperty "ArrowChoice updates only the selected branch" $ H.withTests 1 $ H.property $ do
        let operation previous input = (previous + input, previous + 1)
            pipeline = left (NN.stateful firstLens operation) :: NN.Forward (Int, Bool) (Either Int Int) (Either Int Int)
        NN.forward pipeline (Right 7) (0, True) H.=== (Right 7, (0, True))
        NN.forward pipeline (Left 7) (0, True) H.=== (Left 7, (1, True)),
      testProperty "one stateless definition works as a function and a stateful arrow" $ H.withTests 1 $ H.property $ do
        let parameters = NN.Dense (T.fromList @'[2, 1] [1, 2 :: Double]) 0
            input = T.fromList @'[2] [1, 2 :: Double]
            ordinary = stateless parameters input
            (statefulOutput, state) = NN.forward (stateless parameters) input True
        state H.=== True
        T.toList (T.run T.CPU ordinary) H.=== T.toList (T.run T.CPU statefulOutput),
      testProperty "prepared training threads model state outside the differentiation tape" $ H.withTests 1 $ H.property $ do
        let execute = T.run3 T.CPU $ \parametersInput runningInput featuresInput ->
              AG.backprop (\p -> let (result, next) = NN.forward (predict NN.Train p) (AG.constant featuresInput) runningInput in (T.sum $ result * result, AG.constant next)) parametersInput
            parameters = T.run T.CPU initialParameters
            running = T.run T.CPU initialNetworkState
            features = T.fromListData @'[2, 2] [1, 2, 3, 5 :: Double]
            ((loss, first), gradients) = execute parameters running features
            ((_, second), _) = execute parameters first features
        approxList (T.toList $ NN.batchNormMean $ statistics first) [1, 1.75]
        approxList (T.toList $ NN.batchNormMean $ statistics second) [1.5, 2.625]
        T.toList (marker first) H.=== [17]
        H.assert $ all (\x -> not (isNaN x || isInfinite x)) $ T.toList loss
        H.assert $ any (/= 0) $ T.toList $ NN.denseWeights $ output gradients,
      testProperty "evaluation preserves all state while accepting the same model" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] [1, 2, 3, 5 :: Double]
            (_, next) = NN.forward (predict NN.Eval initialParameters) input initialNetworkState
            (actual, _) = T.randomWords @'[4] $ NN.dropoutGenerator $ randomness next
            (expected, _) = T.randomWords @'[4] $ NN.dropoutGenerator $ randomness initialNetworkState
        T.toList (T.run T.CPU actual) H.=== T.toList (T.run T.CPU expected)
        T.toList (T.run T.CPU $ NN.batchNormMean $ statistics next) H.=== [0, 0],
      testProperty "one prepared step combines model state, gradients and optimizer State" $ H.withTests 1 $ H.property $ do
        let optimizer = NN.Adam 0.01 0.9 0.999 1e-8
            execute = T.run4 T.CPU $ \parametersInput runningInput optimizerInput featuresInput ->
              let ((loss, nextModel), gradients) =
                    AG.backprop
                      (\p -> let (result, next) = NN.forward (predict NN.Train p) (AG.constant featuresInput) runningInput in (T.sum $ result * result, AG.constant next))
                      parametersInput
                  (updated, nextOptimizer) = NN.runState (NN.optimizerStep optimizer parametersInput gradients) optimizerInput
               in (loss, (updated, (nextModel, nextOptimizer)))
            parameters = T.run T.CPU initialParameters
            running = T.run T.CPU initialNetworkState
            optimizerState = T.run T.CPU $ NN.initialState optimizer initialParameters
            features = T.fromListData @'[2, 2] [1, 2, 3, 5 :: Double]
            (_, (firstParameters, (firstModel, firstOptimizer))) = execute parameters running optimizerState features
            (_, (_, (secondModel, (count, _)))) = execute firstParameters firstModel firstOptimizer features
        T.toList count H.=== [2]
        T.toList (marker secondModel) H.=== [17]
        H.assert $ T.toList (NN.denseWeights $ output firstParameters) /= T.toList (NN.denseWeights $ output parameters),
      testProperty "composed lenses focus nested state using ordinary composition" $ H.withTests 1 $ H.property $ do
        let operation previous input = (previous + input, previous + 1)
            pipeline = NN.stateful (firstLens . firstLens) operation
        NN.forward pipeline (3 :: Int) ((2, True), 'x') H.=== (5, ((3, True), 'x')),
      testProperty "optimizer State actions compose multiple updates in do notation" $ H.withTests 1 $ H.property $ do
        let optimizer = NN.Adam 0.1 0.9 0.999 1e-8
            parameters = T.full @'[] (5 :: Double)
            action :: NN.State (NN.OptimizerState NN.Adam (T.Tensor '[] Double)) (T.Tensor '[] Double)
            action = do
              first <- NN.optimizerStep optimizer parameters 2
              NN.optimizerStep optimizer first 3
            (updated, (count, _)) = NN.runState action $ NN.initialState optimizer parameters
        T.toList (T.run T.CPU count) H.=== [2]
        H.assert $ all (< 5) $ T.toList $ T.run T.CPU updated
    ]
