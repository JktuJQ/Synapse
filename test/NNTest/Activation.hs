{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Activation (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "activation"
    [ testProperty "relu preserves positive values and uses zero derivative at zero" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] [-2, 0, 1, 3 :: Double]
            (output, gradient) = AG.backprop NN.relu input
        values (NN.relu input) H.=== [0, 0, 1, 3]
        values output H.=== [0, 0, 1, 3]
        values gradient H.=== [0, 0, 1, 1],
      testProperty "leaky relu uses its supplied slope including at zero" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[4] [-2, 0, 1, 3 :: Double]
            (output, gradient) = AG.backprop (NN.leakyRelu 0.1) input
        approxList (values (NN.leakyRelu 0.1 input)) [-0.2, 0, 1, 3]
        approxList (values output) [-0.2, 0, 1, 3]
        approxList (values gradient) [0.1, 0.1, 1, 1],
      testProperty "leaky relu can differentiate a supplied slope" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [-2, 0, 3 :: Double]
            slope = T.full @'[3] (0.1 :: Double)
            (_, (ds, dx)) = AG.backprop (uncurry NN.leakyRelu) (slope, input)
        values ds H.=== [-2, 0, 0]
        approxList (values dx) [0.1, 0.1, 1],
      testProperty "sigmoid agrees with its formula and derivative including zero" $ H.withTests 1 $ H.property $ do
        let xs = [-2, -0.5, 0, 0.5, 2 :: Double]
            input = T.fromList @'[5] xs
            expected = fmap (\x -> 1 / (1 + exp (-x))) xs
            (output, gradient) = AG.backprop NN.sigmoid input
        approxList (values (NN.sigmoid input)) expected
        approxList (values output) expected
        approxList (values gradient) (fmap (\y -> y * (1 - y)) expected),
      testProperty "tanh agrees with Prelude and differentiates through the existing rule" $ H.withTests 1 $ H.property $ do
        let xs = [-2, -0.5, 0, 0.5, 2 :: Double]
            input = T.fromList @'[5] xs
            expected = fmap tanh xs
            (output, gradient) = AG.backprop NN.tanh input
        approxList (values (NN.tanh input)) expected
        approxList (values output) expected
        approxList (values gradient) (fmap (\y -> 1 - y * y) expected),
      testProperty "softplus agrees with its formula and has a half derivative at zero" $ H.withTests 1 $ H.property $ do
        let xs = [-2, -0.5, 0, 0.5, 2 :: Double]
            input = T.fromList @'[5] xs
            expected = fmap (\x -> log (1 + exp x)) xs
            (output, gradient) = AG.backprop NN.softplus input
        approxList (values (NN.softplus input)) expected
        approxList (values output) expected
        approxList (values gradient) (fmap (\x -> 1 / (1 + exp (-x))) xs),
      testProperty "sigmoid and softplus stay finite for extreme finite inputs" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[4] [-1e200, -1000, 1000, 1e200 :: Double]
            (sigmoidOutput, sigmoidGradient) = AG.backprop NN.sigmoid input
            (softplusOutput, softplusGradient) = AG.backprop NN.softplus input
        values sigmoidOutput H.=== [0, 0, 1, 1]
        values sigmoidGradient H.=== [0, 0, 0, 0]
        values softplusOutput H.=== [0, 0, 1000, 1e200]
        values softplusGradient H.=== [0, 0, 1, 1],
      testProperty "softplus retains small negative-input values and gradients" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (-40 :: Double)
            (output, gradient) = AG.backprop NN.softplus input
            expected = exp (-40 :: Double)
        case (values output, values gradient) of
          ([y], [dy]) -> do
            H.assert (abs (y / expected - 1) < 1e-12)
            H.assert (abs (dy / expected - 1) < 1e-12)
          _ -> H.failure,
      testProperty "activations propagate NaN and handle infinite inputs" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [-1 / 0, 1 / 0, 0 / 0 :: Double]
            check :: T.Tensor '[3] Double -> [Double] -> H.PropertyT IO ()
            check actual expected = case values actual of
              [negative, positive, nan] -> do
                [negative, positive] H.=== expected
                H.assert (isNaN nan)
              _ -> H.failure
        check (NN.relu input) [0, 1 / 0]
        check (NN.leakyRelu 0.1 input) [-1 / 0, 1 / 0]
        check (NN.sigmoid input) [0, 1]
        check (NN.tanh input) [-1, 1]
        check (NN.softplus input) [0, 1 / 0],
      testProperty "activations support Float, empty tensors, and element expressions" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [-100, 0, 100 :: Float]
            empty = T.zeros @'[0, 2] @Double
        approxList (T.toList (T.run T.CPU (NN.sigmoid input))) [0, 0.5, 1]
        approxList (T.toList (T.run T.CPU (NN.softplus input))) [0, log 2, 100]
        values (NN.relu empty) H.=== []
        values (NN.leakyRelu 0.1 empty) H.=== []
        values (NN.sigmoid empty) H.=== []
        values (NN.tanh empty) H.=== []
        values (NN.softplus empty) H.=== []
        let element = T.constant (0 :: Double)
            (output, gradient) = AG.backprop NN.sigmoid element
            (softOutput, softGradient) = AG.backprop NN.softplus element
        values (T.scalar output) H.=== [0.5]
        values (T.scalar gradient) H.=== [0.25]
        approxList (values (T.scalar softOutput)) [log 2]
        approxList (values (T.scalar softGradient)) [0.5]
        values (T.scalar (NN.relu (T.constant (-2 :: Double)))) H.=== [0],
      testProperty "activation composition supports seeded pullbacks and prepared execution" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [-1, 0, 1 :: Double]
            (_, backward) = AG.pullback (NN.sigmoid . NN.relu) input
            gradient = backward (T.fromList @'[3] [2, 3, 4])
            s = 1 / (1 + exp (-1 :: Double))
            execute = T.run1 T.CPU (AG.backprop (T.sum . NN.softplus))
            (output, preparedGradient) = execute (T.run T.CPU input)
        approxList (values gradient) [0, 0, 4 * s * (1 - s)]
        approxList (T.toList output) [log (1 + exp (-1)) + log 2 + log (1 + exp 1)]
        approxList (T.toList preparedGradient) [1 - s, 0.5, s]
    ]
