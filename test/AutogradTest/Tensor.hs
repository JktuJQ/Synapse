{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}

module AutogradTest.Tensor (tests) where

import qualified Hedgehog as H
import qualified Numeric.Backprop.Explicit as BP
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import qualified AutogradTest.Tensor.Elementwise
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

constant :: T.Tensor sh a -> AG.Tensor s sh a
constant = AG.Tensor . AG.Var . BP.auto

differentiate :: (T.Shape from, T.Shape to) => (forall s. (AG.Context s) => AG.Tensor s from Double -> AG.Tensor s to Double) -> T.Tensor from Double -> (T.Tensor to Double, T.Tensor from Double)
differentiate = AG.backprop

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

finiteDifferences :: (T.Shape from, T.Shape to) => (forall s. (AG.Context s) => AG.Tensor s from Double -> AG.Tensor s to Double) -> T.Tensor from Double -> H.PropertyT IO ()
finiteDifferences f input = do
  let xs = values input
      (_, gradient) = differentiate f input
      step = 1e-5
      evaluate ys = sum (values (fst (differentiate f (T.unsafeFromList ys))))
      perturb axis delta = zipWith (\i x -> if axis == i then x + delta else x) [0 ..] xs
      numerical = [(evaluate (perturb axis step) - evaluate (perturb axis (negate step))) / (2 * step) | axis <- [0 .. length xs - 1]]
  approxList (values gradient) numerical

forward :: (T.ShapeOps t Double, T.LinearAlgebra t Double, T.Reductions t Double) => t '[2, 3] Double -> t '[] Double
forward x = T.sum (T.matmul x (T.transpose x))

tests :: TestTree
tests =
  testGroup
    "tensor operations"
    [ AutogradTest.Tensor.Elementwise.tests,
      testProperty "one polymorphic computation supports ordinary tensors and autograd" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (value, gradient) = differentiate forward input
        values value H.=== values (forward input)
        values gradient H.=== [10, 14, 18, 10, 14, 18],
      testProperty "reshape, squeeze and unsqueeze preserve gradient order" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (_, gradient) = differentiate (\x -> T.squeeze @0 (T.unsqueeze @0 (T.reshape @'[3, 2] (x * x)))) input
        values gradient H.=== [2, 4, 6, 8, 10, 12],
      testProperty "axis permutations invert the gradient coordinates" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            weights = T.unsafeFromList [1 .. 6] :: T.Tensor '[3, 2] Double
            (_, gradient) = differentiate (\x -> T.permuteAxes @'[1, 0] x * constant weights) input
        values gradient H.=== [1, 3, 5, 2, 4, 6],
      testProperty "broadcast sums repeated gradient contributions" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[1, 3, 1] Double
            (_, gradient) = differentiate (\x -> T.broadcast @'[2, 3, 4] (x * x)) input
        values gradient H.=== [16, 32, 48],
      testProperty "broadcast to empty dimensions returns zero gradients" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2] :: T.Tensor '[2, 1] Double
            (_, gradient) = differentiate (T.broadcast @'[2, 0]) input
        values gradient H.=== [0, 0],
      testProperty "concat and stack propagate both uses of a shared input" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (_, concatenated) = differentiate (\x -> T.concat @1 x (x * x)) input
            (_, stacked) = differentiate (\x -> T.stack @1 x (x * x)) input
        values concatenated H.=== [3, 5, 7, 9, 11, 13]
        values stacked H.=== [3, 5, 7, 9, 11, 13],
      testProperty "slices scatter gradients through steps, reverse ranges and fixed axes" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 12] :: T.Tensor '[2, 6] Double
            weights = T.unsafeFromList [2, 3, 5] :: T.Tensor '[3] Double
            (_, gradient) = differentiate (\x -> T.slice @'[ 'T.At 1, 'T.RangeReverse 0 6 2] x * constant weights) input
        values gradient H.=== [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 2, 0],
      testProperty "empty slices return zero gradients" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[6] Double
            (_, gradient) = differentiate (T.slice @'[ 'T.Range 4 2 1]) input
        values gradient H.=== replicate 6 0,
      testProperty "index gradients occupy only the selected element" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (value, gradient) =
              BP.backprop
                (BP.ZF AG.zero)
                (BP.OF AG.one)
                (AG.unVar . AG.unTensorElement . T.index @'[1, 2] . AG.Tensor . AG.Var)
                input
        values (T.scalar value) H.=== [6]
        values gradient H.=== [0, 0, 0, 0, 0, 1],
      testProperty "sum and mean expand gradients over the reduced axes" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (_, summed) = differentiate (T.sumAxes @'[1]) input
            (_, averaged) = differentiate (T.meanAxes @'[0]) input
        values summed H.=== replicate 6 1
        values averaged H.=== replicate 6 0.5,
      testProperty "extrema distribute gradients among equal extrema" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 3, 3, 4, 4, 2] :: T.Tensor '[2, 3] Double
            (_, maximumGradient) = differentiate (T.maximumAxes @'[1]) input
            (_, minimumGradient) = differentiate T.minimum input
        values maximumGradient H.=== [0, 0.5, 0.5, 0.5, 0.5, 0]
        values minimumGradient H.=== [1, 0, 0, 0, 0, 0],
      testProperty "normalization pullbacks agree with finite differences" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [0.2, 0.5, 1.1, -0.3, 0.4, 0.8] :: T.Tensor '[2, 3] Double
            weights = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
        finiteDifferences (\x -> T.softmax @1 x * constant weights) input
        finiteDifferences (\x -> T.logSoftmax @1 x * constant weights) input
        finiteDifferences T.logSumExp input
        finiteDifferences (T.logSumExpAxes @'[0]) input,
      testProperty "rectangular and batched matmul pullbacks agree with finite differences" $ H.withTests 1 $ H.property $ do
        let left = T.unsafeFromList [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] :: T.Tensor '[2, 3] Double
            right = T.unsafeFromList [0.7, 0.8, 0.9, 1.0, 1.1, 1.2] :: T.Tensor '[3, 2] Double
            batchLeft = T.unsafeFromList [0.1, 0.2 .. 1.2] :: T.Tensor '[2, 2, 3] Double
            batchRight = T.unsafeFromList [0.2, 0.3 .. 1.3] :: T.Tensor '[2, 3, 2] Double
        finiteDifferences (\x -> T.matmul x (constant right)) left
        finiteDifferences (T.matmul (constant left)) right
        finiteDifferences (\x -> T.matmul x (constant batchRight)) batchLeft
        finiteDifferences (T.matmul (constant batchLeft)) batchRight,
      testProperty "zero-size matrix contractions return zero gradients" $ H.withTests 1 $ H.property $ do
        let left = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            right = T.unsafeFromList [] :: T.Tensor '[3, 0] Double
            (_, gradient) = differentiate (\x -> T.matmul x (constant right)) left
        values gradient H.=== replicate 6 0,
      testProperty "dot accumulates gradients from both vector arguments" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double
            (value, gradient) = differentiate (\x -> T.dot x x) input
        values value H.=== [14]
        values gradient H.=== [2, 4, 6],
      testProperty "empty inner products have empty input gradients" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [] :: T.Tensor '[0] Double
            (value, gradient) = differentiate (\x -> T.dot x x) input
        values value H.=== [0]
        values gradient H.=== []
    ]
