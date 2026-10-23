{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import qualified Hedgehog as H
import qualified NNTest.Activation as Activation
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests = testGroup "nn" [lossTests, Activation.tests]

lossTests :: TestTree
lossTests =
  testGroup
    "nn loss"
    [ testProperty "mse supports partially applied mean, sum, and identity reductions" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            targets = T.fromList @'[2, 2] [1, 0, 5, 1 :: Double]
            meanLoss = NN.mse T.mean
            totalLoss = NN.mse T.sum
            elementLoss = NN.mse id
        values (meanLoss predictions targets) H.=== [4.25]
        values (totalLoss predictions targets) H.=== [17]
        values (elementLoss predictions targets) H.=== [0, 4, 4, 9],
      testProperty "mse accepts axis-wise reductions" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            targets = T.fromList @'[2, 2] [1, 0, 5, 1 :: Double]
        values (NN.mse (T.meanAxes @'[1]) predictions targets) H.=== [2, 6.5],
      testProperty "mse supports scalars and empty unreduced or summed tensors" $ H.withTests 1 $ H.property $ do
        values (NN.mse T.mean (T.full @'[] (3 :: Double)) (T.full @'[] 1)) H.=== [4]
        let empty = T.zeros @'[0, 2] @Double
        values (NN.mse id empty empty) H.=== []
        values (NN.mse T.sum empty empty) H.=== [0],
      testProperty "mse preserves single-precision element types" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2] [1, 3 :: Float]
            targets = T.fromList @'[2] [0, 1 :: Float]
        T.toList (T.run T.CPU (NN.mse T.mean predictions targets)) H.=== [2.5],
      testProperty "mse differentiates predictions and targets with opposite gradients" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            targets = T.fromList @'[2, 2] [1, 0, 5, 1 :: Double]
            (loss, (dp, dt)) = AG.backprop (uncurry (NN.mse T.mean)) (predictions, targets)
        values loss H.=== [4.25]
        approxList (values dp) [0, 1, -1, 1.5]
        approxList (values dt) [0, -1, 1, -1.5],
      testProperty "unreduced mse pullbacks respect nonuniform output seeds" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            targets = T.fromList @'[2, 2] [1, 0, 5, 1 :: Double]
            (_, backward) = AG.pullback (uncurry (NN.mse id)) (predictions, targets)
            (dp, dt) = backward (T.fromList @'[2, 2] [1, 2, 3, 4])
        values dp H.=== [0, 8, -12, 24]
        values dt H.=== [0, -8, 12, -24],
      testProperty "mse accumulates both paths when predictions and targets share a variable" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [1, 2, 3 :: Double]
            (loss, gradient) = AG.backprop (\x -> NN.mse T.mean x x) input
        values loss H.=== [0]
        values gradient H.=== [0, 0, 0],
      testProperty "prepared mse gradients accept changing predictions and targets" $ H.withTests 1 $ H.property $ do
        let execute = T.run2 T.CPU $ \predictions targets ->
              AG.backprop (uncurry (NN.mse T.sum)) (predictions, targets)
            first = T.run T.CPU (T.fromList @'[2] [1, 3 :: Double])
            second = T.run T.CPU (T.fromList @'[2] [0, 1 :: Double])
            (firstLoss, (firstDP, firstDT)) = execute first second
            (secondLoss, (secondDP, secondDT)) = execute second first
        T.toList firstLoss H.=== [5]
        T.toList firstDP H.=== [2, 4]
        T.toList firstDT H.=== [-2, -4]
        T.toList secondLoss H.=== [5]
        T.toList secondDP H.=== [-2, -4]
        T.toList secondDT H.=== [2, 4],
      testProperty "mae supports reductions and differentiates both inputs" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[5] [-3, -1, 0, 1, 3 :: Double]
            targets = T.zeros @'[5] @Double
            (loss, (dp, dt)) = AG.backprop (uncurry (NN.mae T.sum)) (predictions, targets)
        values (NN.mae id predictions targets) H.=== [3, 1, 0, 1, 3]
        approxList (values (NN.mae T.mean predictions targets)) [1.6]
        values loss H.=== [8]
        values dp H.=== [-1, -1, 0, 1, 1]
        values dt H.=== [1, 1, 0, -1, -1],
      testProperty "huber differentiates quadratic, transition, and linear regions" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[7] [-3, -2, -1, 0, 1, 2, 3 :: Double]
            targets = T.zeros @'[7] @Double
            (loss, (dp, dt)) = AG.backprop (uncurry (NN.huber T.sum 2)) (predictions, targets)
        values (NN.huber id 2 predictions targets) H.=== [4, 2, 0.5, 0, 0.5, 2, 4]
        values loss H.=== [13]
        approxList (values dp) [-2, -2, -1, 0, 1, 2, 2]
        approxList (values dt) [2, 2, 1, 0, -1, -2, -2],
      testProperty "smooth L1 normalizes Huber errors and gradients by beta" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[7] [-3, -2, -1, 0, 1, 2, 3 :: Double]
            targets = T.zeros @'[7] @Double
            (loss, (dp, dt)) = AG.backprop (uncurry (NN.smoothL1 T.sum 2)) (predictions, targets)
        values (NN.smoothL1 id 2 predictions targets) H.=== [2, 1, 0.25, 0, 0.25, 1, 2]
        values loss H.=== [6.5]
        approxList (values dp) [-1, -1, -0.5, 0, 0.5, 1, 1]
        approxList (values dt) [1, 1, 0.5, 0, -0.5, -1, -1],
      testProperty "zero beta gives absolute errors and a zero derivative at equality" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[3] [-2, 0, 2 :: Double]
            targets = T.zeros @'[3] @Double
            (loss, gradient) = AG.backprop (\x -> NN.smoothL1 T.sum 0 x (AG.constant targets)) predictions
        values loss H.=== [4]
        values gradient H.=== [-1, 0, 1],
      testProperty "robust losses avoid squaring large outliers" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2] [-1e200, 1e200 :: Double]
            targets = T.zeros @'[2] @Double
            (huberLoss, huberGradient) = AG.backprop (\x -> NN.huber T.sum 1 x (AG.constant targets)) predictions
            (smoothLoss, smoothGradient) = AG.backprop (\x -> NN.smoothL1 T.sum 1 x (AG.constant targets)) predictions
        values huberLoss H.=== [2e200]
        values smoothLoss H.=== [2e200]
        values huberGradient H.=== [-1, 1]
        values smoothGradient H.=== [-1, 1],
      testProperty "new losses support axis reductions, scalars, empty tensors, and Float" $ H.withTests 1 $ H.property $ do
        let predictions = T.fromList @'[2, 2] [0, 1, 2, 3 :: Double]
            targets = T.zeros @'[2, 2] @Double
            empty = T.zeros @'[0] @Double
        values (NN.mae (T.sumAxes @'[1]) predictions targets) H.=== [1, 5]
        values (NN.huber (T.sumAxes @'[1]) 1 predictions targets) H.=== [0.5, 4]
        values (NN.smoothL1 (T.sumAxes @'[1]) 2 predictions targets) H.=== [0.25, 3]
        values (NN.huber id 1 (T.full @'[] (2 :: Double)) (T.full @'[] 0)) H.=== [1.5]
        values (NN.mae id empty empty) H.=== []
        values (NN.huber T.sum 1 empty empty) H.=== [0]
        values (NN.smoothL1 T.sum 1 empty empty) H.=== [0]
        let floatPredictions = T.fromList @'[2] [1, 3 :: Float]
            floatTargets = T.zeros @'[2] @Float
        T.toList (T.run T.CPU (NN.huber T.mean 1 floatPredictions floatTargets)) H.=== [1.5]
        T.toList (T.run T.CPU (NN.smoothL1 T.mean 2 floatPredictions floatTargets)) H.=== [1.125],
      testProperty "robust losses reject invalid transition parameters" $ H.withTests 1 $ H.property $ do
        let input = T.zeros @'[1] @Double :: T.Tensor '[1] Double
            rejects loss threshold = do
              result <- H.evalIO $ try @ErrorCall $ evaluate $ loss threshold input input
              case result of
                Left _ -> pure ()
                Right _ -> H.failure
        mapM_ (rejects (NN.huber T.sum)) [0, -1, 0 / 0, 1 / 0, -1 / 0]
        mapM_ (rejects (NN.smoothL1 T.sum)) [-1, 0 / 0, 1 / 0, -1 / 0],
      testProperty "mse rejects mismatched prediction and target shapes" $ H.withTests 1 $
        Compiler.rejects "test/NNTest/Fixtures/Rejected/MSEShape.hs" "Couldn't match type"
    ]
