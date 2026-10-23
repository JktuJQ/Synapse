{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Metrics (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.NN.Metrics as Metrics
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

checkGradient ::
  (forall s. (AG.Context s) => AG.Tensor s '[3] Double -> AG.Tensor s '[] Double) ->
  H.PropertyT IO ()
checkGradient function = do
  let input = [0.7, 2.2, 4.6 :: Double]
      step = 1e-4
      execute = T.run1 T.CPU (AG.backprop function)
      evaluateAt xs = execute (T.run T.CPU (T.fromList @'[3] xs))
      lossAt xs = case T.toList (fst (evaluateAt xs)) of
        [loss] -> loss
        _ -> error "Expected a scalar metric result"
      perturb axis delta = zipWith (\i x -> if i == axis then x + delta else x) [0 :: Int ..] input
      expected = [(lossAt (perturb axis step) - lossAt (perturb axis (-step))) / (2 * step) | axis <- [0 .. 2]]
      (_, gradient) = evaluateAt input
  approxList (T.toList gradient) expected

tests :: TestTree
tests =
  testGroup
    "metrics"
    [ testProperty "metrics share mse and mae with losses" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[3] [1, 3, 5 :: Double]
            targets = T.fromList @'[3] [0, 1, 3 :: Double]
        values (Metrics.mse T.mean predictions targets) H.=== [3]
        values (Metrics.mae T.sum predictions targets) H.=== [5],
      testProperty "rmse supports full and axis-wise reductions" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            targets = T.fromList @'[2, 2] [1, 0, 5, 1 :: Double]
        approxList (values (NN.rmse T.mean predictions targets)) [sqrt 4.25]
        approxList (values (NN.rmse (T.meanAxes @'[1]) predictions targets)) [sqrt 2, sqrt 6.5]
        approxList (values (NN.rmse id predictions targets)) [0, 2, 2, 3],
      testProperty "rmse handles perfect, scalar, empty, and Float inputs" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [1, 3 :: Double]
            empty = T.zeros @'[0] @Double
        values (NN.rmse T.mean input input) H.=== [0]
        values (NN.rmse T.mean (T.full @'[] (3 :: Double)) (T.full @'[] 1)) H.=== [2]
        values (NN.rmse id empty empty) H.=== []
        H.assert (all isNaN (values (NN.rmse T.mean empty empty)))
        let floatInput = T.fromList @'[2] [1, 3 :: Float]
        approxList (T.toList (T.run T.CPU (NN.rmse T.mean floatInput (T.zeros @'[2])))) [sqrt 5],
      testProperty "r2 measures perfect, mean-baseline, and worse predictions" $ H.withTests 1 $ H.property $ do
        let targets = T.fromList @'[3] [1, 2, 3 :: Double]
        values (NN.r2 targets targets) H.=== [1]
        values (NN.r2 (T.full @'[3] 2) targets) H.=== [0]
        values (NN.r2 (T.zeros @'[3]) targets) H.=== [-6],
      testProperty "r2 agrees with a reference regression example" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[4] [2.5, 0, 2, 8 :: Double]
            targets = T.fromList @'[4] [3, -0.5, 2, 7 :: Double]
        approxList (values (NN.r2 predictions targets)) [1 - 1.5 / 29.1875],
      testProperty "r2 uses a global mean for multidimensional tensors" $ H.withTests 1 $ H.property $ do
        let targets = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            predictions = T.fromList @'[2, 2] [2, 2, 2, 2 :: Double]
        approxList (values (NN.r2 predictions targets)) [-0.2]
        let floatTargets = T.fromList @'[3] [1, 2, 3 :: Float]
        T.toList (T.run T.CPU (NN.r2 floatTargets floatTargets)) H.=== [1],
      testProperty "r2 keeps undefined constant-target scores explicit" $ H.withTests 1 $ H.property $ do
        let targets = T.full @'[3] (2 :: Double)
        H.assert (all isNaN (values (NN.r2 targets targets)))
        values (NN.r2 (T.zeros @'[3]) targets) H.=== [-(1 / 0)],
      testProperty "r2 rejects fewer than two observations" $ H.withTests 1 $ H.property $ do
        let scalar = T.full @'[] (1 :: Double)
            singleton = T.full @'[1] (1 :: Double)
            empty = T.zeros @'[0] @Double
            rejects :: T.Tensor '[] Double -> H.PropertyT IO ()
            rejects score = do
              result <- H.evalIO $ try @ErrorCall $ evaluate score
              case result of
                Left _ -> pure ()
                Right _ -> H.failure
        rejects (NN.r2 scalar scalar)
        rejects (NN.r2 singleton singleton)
        rejects (NN.r2 empty empty),
      testProperty "regression metrics remain usable with autograd" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[3] [0, 2, 4 :: Double]
            targets = T.fromList @'[3] [1, 2, 3 :: Double]
            (rootError, rootGradient) = AG.backprop (\p -> NN.rmse T.mean p (AG.constant targets)) predictions
            (score, (dp, dt)) = AG.backprop (uncurry NN.r2) (predictions, targets)
            root = sqrt (2 / 3)
        approxList (values rootError) [root]
        approxList (values rootGradient) [-(1 / (3 * root)), 0, 1 / (3 * root)]
        values score H.=== [0]
        approxList (values dp) [1, 0, -1]
        approxList (values dt) [-2, 0, 2],
      testProperty "signed, maximum, and percentage errors agree with reference values" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[3] [2, 2, 5 :: Double]
            targets = T.fromList @'[3] [1, 2, 4 :: Double]
        approxList (values (NN.meanError T.mean predictions targets)) [2 / 3]
        values (NN.meanError id predictions targets) H.=== [1, 0, 1]
        values (NN.maxError predictions targets) H.=== [1]
        approxList (values (NN.mape T.mean 1e-6 predictions targets)) [5 / 12]
        approxList (values (NN.smape T.mean 1e-6 predictions targets)) [8 / 27]
        approxList (values (NN.wmape 1e-6 predictions targets)) [2 / 7],
      testProperty "percentage errors stabilize zero targets with an explicit epsilon" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2] [0, 1 :: Double]
            targets = T.zeros @'[2] @Double
        values (NN.mape T.mean 0.5 predictions targets) H.=== [1]
        values (NN.smape T.mean 0.5 predictions targets) H.=== [1]
        values (NN.wmape 0.5 predictions targets) H.=== [2]
        values (NN.mape T.mean 0.5 targets targets) H.=== [0]
        values (NN.smape T.mean 0.5 targets targets) H.=== [0]
        values (NN.wmape 0.5 targets targets) H.=== [0],
      testProperty "percentage errors support negative targets and smape is symmetric" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2] [-2, 4 :: Double]
            targets = T.fromList @'[2] [-1, -2 :: Double]
        values (NN.mape id 0.1 predictions targets) H.=== [1, 3]
        approxList (values (NN.smape T.mean 0.1 predictions targets)) [4 / 3]
        values (NN.smape T.mean 0.1 predictions targets) H.=== values (NN.smape T.mean 0.1 targets predictions)
        approxList (values (NN.wmape 0.1 predictions targets)) [7 / 3],
      testProperty "logarithmic errors support reductions and preserve their domain" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2] [0, 3 :: Double]
            targets = T.ones @'[2] @Double
            squaredLog = log 2 * log 2
        approxList (values (NN.msle id predictions targets)) [squaredLog, squaredLog]
        approxList (values (NN.msle T.mean predictions targets)) [squaredLog]
        approxList (values (NN.rmsle T.mean predictions targets)) [log 2]
        values (NN.msle T.mean targets targets) H.=== [0]
        H.assert (all isNaN (values (NN.msle id (T.full @'[] (-2 :: Double)) (T.zeros @'[]))))
        values (NN.msle id (T.full @'[] (-1 :: Double)) (T.zeros @'[])) H.=== [1 / 0],
      testProperty "relative errors and explained variance agree with reference values" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[3] [2, 2, 5 :: Double]
            targets = T.fromList @'[3] [1, 2, 4 :: Double]
        approxList (values (NN.rae predictions targets)) [0.6]
        approxList (values (NN.rse predictions targets)) [3 / 7]
        approxList (values (NN.explainedVariance predictions targets)) [6 / 7]
        approxList (values (NN.pearson predictions targets)) [5 / sqrt 28]
        approxList (values (NN.r2 predictions targets + NN.rse predictions targets)) [1]
        values (NN.rae targets targets) H.=== [0]
        values (NN.rse targets targets) H.=== [0],
      testProperty "explained variance and correlation ignore constant offsets" $ H.withTests 1 $ H.property $ do
        let targets = T.fromList @'[3] [1, 2, 4 :: Double]
        approxList (values (NN.explainedVariance (targets + 10) targets)) [1]
        approxList (values (NN.pearson (2 * targets + 10) targets)) [1]
        approxList (values (NN.pearson (10 - 2 * targets) targets)) [-1]
        approxList (values (NN.pearson targets (2 * targets + 10))) [1],
      testProperty "undefined variance and relative-error denominators remain nonfinite" $ H.withTests 1 $ H.property $ do
        let targets = T.full @'[3] (2 :: Double)
            predictions = T.fromList @'[3] [1, 2, 3 :: Double]
        H.assert (all isNaN (values (NN.rae targets targets)))
        H.assert (all isNaN (values (NN.rse targets targets)))
        H.assert (all isNaN (values (NN.explainedVariance targets targets)))
        H.assert (all isNaN (values (NN.pearson predictions targets)))
        H.assert (all isNaN (values (NN.pearson targets predictions)))
        values (NN.rae predictions targets) H.=== [1 / 0]
        values (NN.rse predictions targets) H.=== [1 / 0]
        values (NN.explainedVariance predictions targets) H.=== [-(1 / 0)],
      testProperty "new metrics support empty, scalar, axis-wise, and Float inputs" $ H.withTests 1 $ H.property $ do
        let empty = T.zeros @'[0] @Double
            predictions = T.fromList @'[2, 2] [2, 4, 6, 8 :: Double]
            targets = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
        values (NN.mape (T.meanAxes @'[1]) 0.1 predictions targets) H.=== [1, 1]
        approxList (values (NN.smape (T.meanAxes @'[1]) 0.1 predictions targets)) [2 / 3, 2 / 3]
        values (NN.meanError (T.sumAxes @'[1]) predictions targets) H.=== [3, 7]
        values (NN.mape id 0.1 empty empty) H.=== []
        values (NN.msle id empty empty) H.=== []
        values (NN.wmape 0.1 empty empty) H.=== [0]
        H.assert (all isNaN (values (NN.smape T.mean 0.1 empty empty)))
        values (NN.maxError (T.full @'[] (3 :: Double)) (T.full @'[] 1)) H.=== [2]
        let floatInput = T.fromList @'[3] [1, 2, 4 :: Float]
            runFloat = T.toList . T.run T.CPU :: T.Tensor '[] Float -> [Float]
        approxList (runFloat (NN.mape T.mean 0.1 (2 * floatInput) floatInput)) [1]
        approxList (runFloat (NN.pearson floatInput floatInput)) [1]
        approxList (runFloat (NN.rmsle T.mean floatInput floatInput)) [0],
      testProperty "percentage metrics reject nonpositive and nonfinite epsilon" $ H.withTests 1 $ H.property $ do
        let input = T.ones @'[2] @Double :: T.Tensor '[2] Double
            rejects :: T.Tensor '[] Double -> H.PropertyT IO ()
            rejects score = do
              result <- H.evalIO $ try @ErrorCall $ evaluate score
              case result of
                Left _ -> pure ()
                Right _ -> H.failure
        mapM_ (\epsilon -> rejects (NN.mape T.mean epsilon input input)) [0, -1, 0 / 0, 1 / 0, -(1 / 0)]
        mapM_ (\epsilon -> rejects (NN.smape T.mean epsilon input input)) [0, -1, 0 / 0, 1 / 0, -(1 / 0)]
        mapM_ (\epsilon -> rejects (NN.wmape epsilon input input)) [0, -1, 0 / 0, 1 / 0, -(1 / 0)],
      testProperty "global variance metrics reject undersized inputs" $ H.withTests 1 $ H.property $ do
        let input = T.ones @'[1] @Double :: T.Tensor '[1] Double
            empty = T.zeros @'[0] @Double :: T.Tensor '[0] Double
            rejects :: T.Tensor '[] Double -> H.PropertyT IO ()
            rejects score = do
              result <- H.evalIO $ try @ErrorCall $ evaluate score
              case result of
                Left _ -> pure ()
                Right _ -> H.failure
        rejects (NN.rae input input)
        rejects (NN.rse input input)
        rejects (NN.explainedVariance input input)
        rejects (NN.pearson input input)
        rejects (NN.rae empty empty)
        rejects (NN.rse empty empty)
        rejects (NN.explainedVariance empty empty),
      testProperty "new metric gradients agree with finite differences" $ H.withTests 1 $ H.property $ do
        let targets = T.fromList @'[3] [1, 2, 4 :: Double]
        checkGradient (\p -> NN.meanError T.mean p (AG.constant targets))
        checkGradient (\p -> NN.maxError p (AG.constant targets))
        checkGradient (\p -> NN.mape T.mean 0.1 p (AG.constant targets))
        checkGradient (\p -> NN.smape T.mean 0.1 p (AG.constant targets))
        checkGradient (\p -> NN.wmape 0.1 p (AG.constant targets))
        checkGradient (\p -> NN.msle T.mean p (AG.constant targets))
        checkGradient (\p -> NN.rmsle T.mean p (AG.constant targets))
        checkGradient (\p -> NN.rae p (AG.constant targets))
        checkGradient (\p -> NN.rse p (AG.constant targets))
        checkGradient (\p -> NN.explainedVariance p (AG.constant targets))
        checkGradient (\p -> NN.pearson p (AG.constant targets)),
      testProperty "maximum error rejects empty tensors" $
        H.withTests 1 $
          Compiler.rejects "test/NNTest/Fixtures/Rejected/MaxErrorEmpty.hs" "Cannot reduce empty tensor",
      testProperty "pearson rejects empty tensors" $
        H.withTests 1 $
          Compiler.rejects "test/NNTest/Fixtures/Rejected/PearsonEmpty.hs" "Cannot reduce empty tensor",
      testProperty "prepared metrics accept changing predictions" $ H.withTests 1 $ H.property $ do
        let execute = T.run2 T.CPU $ \predictions targets ->
              (NN.rmse T.mean predictions targets, NN.r2 predictions targets)
            targetData = T.run T.CPU (T.fromList @'[3] [1, 2, 3 :: Double])
            baseline = T.run T.CPU (T.full @'[3] (2 :: Double))
            (perfectError, perfectScore) = execute targetData targetData
            (baselineError, baselineScore) = execute baseline targetData
        T.toList perfectError H.=== [0]
        T.toList perfectScore H.=== [1]
        approxList (T.toList baselineError) [sqrt (2 / 3)]
        T.toList baselineScore H.=== [0]
    ]
