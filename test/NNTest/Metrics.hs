{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Metrics (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.NN.Metrics as Metrics
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "metrics"
    [ testProperty "multiclass scores distinguish micro, macro and support weighting" $ H.withTests 1 $ H.property $ do
        let logits = T.fromList @'[6, 3] [4, 0, 0, 4, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4, 0, 0, 4] :: T.Tensor '[6, 3] Double
            targets = T.fromList @'[6] [0, 0, 0, 1, 1, 2]
        approxList (values $ NN.precisionPerClass @1 logits targets) [1, 0.5, 0.5]
        approxList (values $ NN.recallPerClass @1 logits targets) [2 / 3, 0.5, 1]
        approxList (values $ NN.f1PerClass @1 logits targets) [0.8, 0.5, 2 / 3]
        approxList (values $ NN.precision @1 NN.Micro logits targets) [2 / 3]
        approxList (values $ NN.recall @1 NN.Micro logits targets) [2 / 3]
        approxList (values $ NN.f1 @1 NN.Micro logits targets) [2 / 3]
        approxList (values $ NN.precision @1 NN.Macro logits targets) [2 / 3]
        approxList (values $ NN.recall @1 NN.Macro logits targets) [13 / 18]
        approxList (values $ NN.f1 @1 NN.Macro logits targets) [59 / 90]
        approxList (values $ NN.precision @1 NN.Weighted logits targets) [0.75]
        approxList (values $ NN.recall @1 NN.Weighted logits targets) [2 / 3]
        approxList (values $ NN.f1 @1 NN.Weighted logits targets) [61 / 90]
        T.toList (T.run T.CPU $ NN.confusionMatrix @1 logits targets) H.=== [2, 1, 0, 0, 1, 1, 0, 0, 1],
      testProperty "classification metrics preserve axis conventions and probability predictions" $ H.withTests 1 $ H.property $ do
        let logits = T.fromList @'[3, 3] [4, 0, 0, 0, 4, 0, 0, 4, 0] :: T.Tensor '[3, 3] Double
            targets = T.fromList @'[3] [0, 1, 2]
        approxList (values $ NN.precision @0 NN.Macro (T.transpose logits) targets) [0.5]
        approxList (values $ NN.recall @1 NN.Macro (T.softmax @1 logits) targets) [2 / 3]
        approxList (values $ NN.f1PerClass @1 logits targets) [1, 2 / 3, 0]
        T.toList (T.run T.CPU $ NN.confusionMatrix @0 (T.transpose logits) targets) H.=== [1, 0, 0, 0, 1, 0, 0, 1, 0],
      testProperty "absent classes contribute zero to macro and no weight to weighted scores" $ H.withTests 1 $ H.property $ do
        let logits = T.fromList @'[2, 4] [3, 0, 0, 0, 0, 3, 0, 0] :: T.Tensor '[2, 4] Double
            targets = T.fromList @'[2] [0, 1]
        values (NN.precisionPerClass @1 logits targets) H.=== [1, 1, 0, 0]
        values (NN.recall @1 NN.Macro logits targets) H.=== [0.5]
        values (NN.f1 @1 NN.Weighted logits targets) H.=== [1]
        values (NN.f1 @1 NN.Micro logits targets) H.=== [1],
      testProperty "empty observations produce zero class scores and confusion counts" $ H.withTests 1 $ H.property $ do
        let logits = T.fromList @'[0, 3] [] :: T.Tensor '[0, 3] Double
            targets = T.fromList @'[0] []
        values (NN.precisionPerClass @1 logits targets) H.=== [0, 0, 0]
        values (NN.recallPerClass @1 logits targets) H.=== [0, 0, 0]
        values (NN.f1PerClass @1 logits targets) H.=== [0, 0, 0]
        values (NN.precision @1 NN.Micro logits targets) H.=== [0]
        values (NN.recall @1 NN.Macro logits targets) H.=== [0]
        values (NN.f1 @1 NN.Weighted logits targets) H.=== [0]
        T.toList (T.run T.CPU $ NN.confusionMatrix @1 logits targets) H.=== replicate 9 0,
      testProperty "invalid target labels cannot index outside class statistics" $ H.withTests 1 $ H.property $ do
        let logits = T.full @'[4, 3] 0 :: T.Tensor '[4, 3] Double
            targets = T.fromList @'[4] [minBound, -1, 3, maxBound]
        values (NN.precision @1 NN.Micro logits targets) H.=== [0]
        values (NN.f1PerClass @1 logits targets) H.=== [0, 0, 0]
        T.toList (T.run T.CPU $ NN.confusionMatrix @1 logits targets) H.=== replicate 9 0,
      testProperty "binary precision recall and F1 count the positive label" $ H.withTests 1 $ H.property $ do
        let logits = T.fromList @'[6] [1, 2, -1, 3, 4, -2] :: T.Tensor '[6] Double
            targets = T.fromList @'[6] [1, 1, 1, 0, 0, 0]
        values (NN.binaryPrecision 0 logits targets) H.=== [0.5]
        approxList (values $ NN.binaryRecall 0 logits targets) [2 / 3]
        approxList (values $ NN.binaryF1 0 logits targets) [4 / 7]
        approxList (values $ NN.binaryF1 0.5 (NN.sigmoid logits) targets) [4 / 7],
      testProperty "binary scores define zero denominators thresholds and empty observations" $ H.withTests 1 $ H.property $ do
        let empty = T.fromList @'[0] [] :: T.Tensor '[0] Double
            zero = T.zeros @'[3] :: T.Tensor '[3] Double
        values (NN.binaryPrecision 0.5 zero zero) H.=== [0]
        values (NN.binaryRecall 0 zero zero) H.=== [0]
        values (NN.binaryF1 0.5 empty empty) H.=== [0]
        values (NN.binaryPrecision 0 zero (T.ones @'[3])) H.=== [1],
      testProperty "class metrics handle unbatched and spatial observations" $ H.withTests 1 $ H.property $ do
        let unbatched = T.fromList @'[3] [0, 0, 0] :: T.Tensor '[3] Double
            spatial = T.fromList @'[1, 3, 2] [3, 0, 0, 3, 0, 0] :: T.Tensor '[1, 3, 2] Double
        values (NN.precisionPerClass @0 unbatched (T.full @'[] 0)) H.=== [1, 0, 0]
        values (NN.f1 @1 NN.Micro spatial (T.fromList @'[1, 2] [0, 1])) H.=== [1],
      testProperty "class metrics accept differentiable predictions without recording gradients" $ H.withTests 1 $ H.property $ do
        let logits = T.fromList @'[2, 3] [3, 0, 0, 0, 3, 0] :: T.Tensor '[2, 3] Double
            targets = T.fromList @'[2] [0, 1]
            (score, gradient) = AG.backprop (\x -> AG.constant $ NN.f1 @1 NN.Micro x targets) logits
        values score H.=== [1]
        values gradient H.=== replicate 6 0,
      testProperty "prepared class metrics accept changing inputs and preserve Float" $ H.withTests 1 $ H.property $ do
        let execute = T.run2 T.CPU $ \predictions targets -> (NN.f1 @1 NN.Micro (predictions :: T.Tensor '[2, 2] Float) targets, NN.confusionMatrix @1 predictions targets)
            logits = T.fromListData @'[2, 2] [3, 0, 0, 3 :: Float]
            (perfect, matrix) = execute logits $ T.fromListData @'[2] [0, 1]
            (wrong, _) = execute logits $ T.fromListData @'[2] [1, 0]
        T.toList perfect H.=== [1]
        T.toList wrong H.=== [0]
        T.toList matrix H.=== [1, 0, 0, 1],
      testProperty "metrics share mse and mae with losses" $ H.withTests 1 $ H.property $ do
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
