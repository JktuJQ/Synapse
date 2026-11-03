{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module NNTest.Regression (tests) where

import Control.Monad (forM_)
import Control.Monad.Trans.State.Strict (evalState)
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import System.Random (mkStdGen)
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

newtype Regression t a = Regression {regressionLayer :: NN.Dense 1 1 t a}
  deriving (Generic)

type instance T.WithTensor t (Regression T.Tensor Double) = Regression t Double

forward ::
  (T.TensorOps t Double, Num (t '[4, 1] Double)) =>
  Regression t Double ->
  t '[4, 1] Double ->
  t '[4, 1] Double
forward parameters = NN.dense (regressionLayer parameters)

objective ::
  (AG.Context s) =>
  T.Tensor '[4, 1] Double ->
  T.Tensor '[4, 1] Double ->
  Regression (AG.Tensor s) Double ->
  AG.Tensor s '[] Double
objective inputs targets parameters =
  NN.mse T.mean (forward parameters (AG.constant inputs)) (AG.constant targets)

step ::
  NN.SGD ->
  Regression T.Tensor Double ->
  T.Tensor '[4, 1] Double ->
  T.Tensor '[4, 1] Double ->
  (T.Tensor '[] Double, Regression T.Tensor Double)
step optimizer parameters inputs targets =
  let (loss, gradients) = AG.backprop (objective inputs targets) parameters
      updated = evalState (NN.optimizerStep optimizer parameters gradients) (NN.initialState optimizer parameters)
   in (loss, updated)

prepareStep ::
  NN.SGD ->
  Regression T.TensorData Double ->
  T.TensorData '[4, 1] Double ->
  T.TensorData '[4, 1] Double ->
  (T.TensorData '[] Double, Regression T.TensorData Double)
prepareStep optimizer = T.run3 T.CPU (step optimizer)

materialize :: Regression T.Tensor Double -> Regression T.TensorData Double
materialize = T.run T.CPU

initialParameters :: Regression T.TensorData Double
initialParameters =
  materialize $
    Regression $
      NN.Dense
        (NN.xavierUniform @'[1, 1] 1 1 1 (mkStdGen 19))
        (NN.uniform @'[1] (-0.25) 0.25 (mkStdGen 23))

trainingInputs :: T.TensorData '[4, 1] Double
trainingInputs = T.run T.CPU (T.fromList @'[4, 1] [-1.5, -0.5, 0.5, 1.5])

trainingTargets :: T.TensorData '[4, 1] Double
trainingTargets = T.run T.CPU (T.fromList @'[4, 1] [-4, -2, 0, 2])

history ::
  Int ->
  (Regression T.TensorData Double -> T.TensorData '[4, 1] Double -> T.TensorData '[4, 1] Double -> (T.TensorData '[] Double, Regression T.TensorData Double)) ->
  Regression T.TensorData Double ->
  [(T.TensorData '[] Double, Regression T.TensorData Double)]
history count execute parameters
  | count <= 0 = []
  | otherwise =
      let current@(_, updated) = execute parameters trainingInputs trainingTargets
       in current : history (count - 1) execute updated

coefficients :: Regression T.TensorData Double -> [Double]
coefficients (Regression parameters) = T.toList (NN.denseWeights parameters) ++ T.toList (NN.denseBias parameters)

tests :: TestTree
tests =
  testGroup
    "regression training"
    [ testProperty "materialized training steps follow the analytic full-batch SGD trajectory" $ H.withTests 1 $ H.property $ do
        let execute = prepareStep (NN.SGD 0.2)
            parameters = initialParameters
            snapshots = history 6 execute parameters
        case coefficients parameters of
          [initialWeight, initialBias] ->
            forM_ (zip [1 :: Int ..] snapshots) $ \(iteration, (loss, updated)) -> do
              let previous = iteration - 1
                  expectedWeight = 2 + (initialWeight - 2) * 0.5 ^ iteration
                  expectedBias = -1 + (initialBias + 1) * 0.6 ^ iteration
                  expectedLoss = 1.25 * (initialWeight - 2) ^ (2 :: Int) * 0.25 ^ previous + (initialBias + 1) ^ (2 :: Int) * 0.36 ^ previous
              approxList (T.toList loss) [expectedLoss]
              approxList (coefficients updated) [expectedWeight, expectedBias]
          _ -> H.failure,
      testProperty "initialized dense regression learns its coefficients and predicts unseen inputs" $ H.withTests 1 $ H.property $ do
        let execute = prepareStep (NN.SGD 0.2)
            snapshots = history 32 execute initialParameters
            predict = T.run3 T.CPU $ \weights bias inputs -> forward (Regression (NN.Dense weights bias)) inputs
            unseen = T.run T.CPU (T.fromList @'[4, 1] [-2, 0, 2, 4 :: Double])
        case reverse snapshots of
          (lastLoss, learned@(Regression parameters)) : _ -> do
            approxList (coefficients learned) [2, -1]
            H.assert (all (< 1e-10) (T.toList lastLoss))
            let predictions = predict (NN.denseWeights parameters) (NN.denseBias parameters) unseen
            approxList (T.toList predictions) [-5, -1, 3, 7]
            H.assert (and (zipWith (>) (concatMap (T.toList . fst) snapshots) (concatMap (T.toList . fst) (drop 1 snapshots))))
          _ -> H.failure,
      testProperty "one prepared training function accepts updated parameters and changing data and targets" $ H.withTests 1 $ H.property $ do
        let execute = prepareStep (NN.SGD 0.2)
            initial = materialize (Regression (NN.Dense 0 0))
            (firstLoss, first) = execute initial trainingInputs trainingTargets
            otherInputs = T.run T.CPU (T.fromList @'[4, 1] [-3, -1, 1, 3 :: Double])
            otherTargets = T.run T.CPU (T.fromList @'[4, 1] [5, 3, 1, -1 :: Double])
            (secondLoss, second) = execute first otherInputs otherTargets
        approxList (T.toList firstLoss) [6]
        approxList (coefficients first) [1, -0.4]
        approxList (T.toList secondLoss) [25.76]
        approxList (coefficients second) [-3, 0.56]
    ]
