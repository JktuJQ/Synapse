{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeApplications #-}

module AutogradTest.Tensor (tests) where

import qualified AutogradTest.Tensor.Elementwise
import qualified AutogradTest.Tensor.Lifting
import qualified Data.Array.Accelerate as A
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Type as Internal
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

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

forward :: (T.TensorOps t Double) => t '[2, 3] Double -> t '[] Double
forward x = T.sum (T.matmul x (T.transpose x))

tests :: TestTree
tests =
  testGroup
    "tensor operations"
    [ AutogradTest.Tensor.Lifting.tests,
      AutogradTest.Tensor.Elementwise.tests,
      testProperty "one polymorphic computation supports ordinary tensors and autograd" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (value, gradient) = differentiate forward input
        values value H.=== values (forward input)
        values gradient H.=== [10, 14, 18, 10, 14, 18],
      testProperty "construction uses the same API inside a differentiation scope" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            generated = T.fromGen @'[2, 3] (\ix -> T.fromIntegral (T.coordinate @0 ix * 3 + T.coordinate @1 ix))
            (_, gradient) = differentiate (\x -> T.sum (x * T.fromGen @'[2, 3] (\ix -> T.fromIntegral (T.coordinate @0 ix * 3 + T.coordinate @1 ix)))) input
        values gradient H.=== values generated,
      testProperty "host conversions construct constants in differentiation scopes" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [1, 2, 3 :: Double]
            host = T.run T.CPU input
            (value, gradient) = differentiate (\x -> x * T.fromList [2, 3, 4] + T.unsafeFromList [5, 6, 7] + T.fromData host) input
            (_, constantGradient) = differentiate (\_ -> T.fromList @'[3] [2, 3, 4] + T.unsafeFromList [5, 6, 7] + T.fromData host) input
        values value H.=== [8, 14, 22]
        values gradient H.=== [2, 3, 4]
        values constantGradient H.=== [0, 0, 0],
      testProperty "like constructors preserve shape without adding a gradient edge" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (value, gradient) = differentiate (\x -> T.sum (x * T.fullLike x 3 + T.onesLike x + T.zerosLike x)) input
        values value H.=== [69]
        values gradient H.=== replicate 6 3,
      testProperty "reshape, squeeze and unsqueeze preserve gradient order" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (_, gradient) = differentiate (\x -> T.squeeze @0 (T.unsqueeze @0 (T.reshape @'[3, 2] (x * x)))) input
        values gradient H.=== [2, 4, 6, 8, 10, 12],
      testProperty "axis permutations invert the gradient coordinates" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            weights = T.unsafeFromList [1 .. 6] :: T.Tensor '[3, 2] Double
            (_, gradient) = differentiate (\x -> T.permuteAxes @'[1, 0] x * AG.constant weights) input
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
            (_, gradient) = differentiate (\x -> T.slice @'[ 'T.At 1, 'T.RangeReverse 0 6 2] x * AG.constant weights) input
        values gradient H.=== [0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 2, 0],
      testProperty "empty slices return zero gradients" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[6] Double
            (_, gradient) = differentiate (T.slice @'[ 'T.Range 4 2 1]) input
        values gradient H.=== replicate 6 0,
      testProperty "index gradients occupy only the selected element" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (value, gradient) = differentiate (T.scalar . T.index @'[1, 2]) input
        values value H.=== [6]
        values gradient H.=== [0, 0, 0, 0, 0, 1],
      testProperty "gather accumulates repeated selections and respects output seeds" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 3] [1 .. 6 :: Double]
            indices = T.fromList @'[2, 3] [2, 2, 0, 1, 0, 1]
            (_, backward) = AG.pullback (\x -> T.unsafeGather @1 x indices) input
            gradient = backward (T.fromList @'[2, 3] [1 .. 6])
            (_, checked) = differentiate (\x -> T.gatherOr @1 0 x indices) input
        values gradient H.=== [3, 0, 3, 5, 10, 0]
        values checked H.=== [1, 0, 2, 1, 2, 0],
      testProperty "safe gather discards invalid gradients and supports empty source axes" $ H.withTests 1 $ H.property $ do
        let indices = T.fromList @'[2, 3] [-1, 3, 0, 1, maxBound, minBound]
            (result, gradient) = differentiate (\x -> T.gatherOr @1 9 x indices) (T.fromList @'[2, 3] [1 .. 6 :: Double])
            (emptyResult, emptyGradient) = differentiate (\x -> T.gatherOr @1 9 x indices) (T.fromList @'[2, 0] [] :: T.Tensor '[2, 0] Double)
        values result H.=== [9, 9, 1, 5, 9, 9]
        values gradient H.=== [1, 0, 0, 0, 1, 0]
        values emptyResult H.=== replicate 6 9
        values emptyGradient H.=== [],
      testProperty "dynamic index gradients reject invalid coordinate aliases" $ H.withTests 1 $ H.property $ do
        let selected row col = Internal.TensorIndex (A.index2 (A.constant row) (A.constant col))
            input = T.fromList @'[2, 3] [1 .. 6 :: Double]
            (_, valid) = differentiate (\x -> T.scalar (T.unsafeDynamicIndex x (selected 1 0))) input
            (fallback, invalid) = differentiate (\x -> T.scalar (T.dynamicIndexOr 7 x (selected 0 3))) input
        values valid H.=== [0, 0, 0, 1, 0, 0]
        values fallback H.=== [7]
        values invalid H.=== replicate 6 0,
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
      testProperty "arg extrema return ordinary indices that can be reshaped" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 3] [1, 5, 2, 4, 3, 6 :: Double]
            (maxima, minima) = T.run T.CPU $ AG.evaluate (\x -> AG.constant (T.reshape @'[2, 1] (T.argmax @1 x), T.reshape @'[2, 1] (T.argmin @1 x))) input
        T.toList maxima H.=== [1, 2]
        T.toList minima H.=== [0, 1],
      testProperty "arg extrema can supply constants without adding gradient edges" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 3] [1, 5, 2, 4, 3, 6 :: Double]
            function x =
              let indices = T.reshape @'[2, 1] (T.argmax @1 x + T.argmin @1 x)
                  weights = T.broadcast @'[2, 3] (T.fromIntegral indices)
               in x * AG.constant weights
            (output, gradient) = differentiate function input
        values output H.=== [1, 5, 2, 12, 9, 18]
        values gradient H.=== [1, 1, 1, 3, 3, 3],
      testProperty "normalization pullbacks agree with finite differences" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [0.2, 0.5, 1.1, -0.3, 0.4, 0.8] :: T.Tensor '[2, 3] Double
            weights = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
        finiteDifferences (\x -> T.softmax @1 x * AG.constant weights) input
        finiteDifferences (\x -> T.logSoftmax @1 x * AG.constant weights) input
        finiteDifferences T.logSumExp input
        finiteDifferences (T.logSumExpAxes @'[0]) input,
      testProperty "rectangular and batched matmul pullbacks agree with finite differences" $ H.withTests 1 $ H.property $ do
        let left = T.unsafeFromList [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] :: T.Tensor '[2, 3] Double
            right = T.unsafeFromList [0.7, 0.8, 0.9, 1.0, 1.1, 1.2] :: T.Tensor '[3, 2] Double
            batchLeft = T.unsafeFromList [0.1, 0.2 .. 1.2] :: T.Tensor '[2, 2, 3] Double
            batchRight = T.unsafeFromList [0.2, 0.3 .. 1.3] :: T.Tensor '[2, 3, 2] Double
        finiteDifferences (\x -> T.matmul x (AG.constant right)) left
        finiteDifferences (T.matmul (AG.constant left)) right
        finiteDifferences (\x -> T.matmul x (AG.constant batchRight)) batchLeft
        finiteDifferences (T.matmul (AG.constant batchLeft)) batchRight,
      testProperty "vector matmul pullbacks agree with finite differences" $ H.withTests 1 $ H.property $ do
        let vector = T.fromList @'[3] [0.2, 0.4, 0.6 :: Double]
            other = T.fromList @'[3] [0.3, 0.5, 0.7 :: Double]
            matrix = T.fromList @'[2, 3] [0.1, 0.2 .. 0.6 :: Double]
            right = T.fromList @'[3, 2] [0.2, 0.3 .. 0.7 :: Double]
            batches = T.fromList @'[2, 3, 2] [0.1, 0.2 .. 1.2 :: Double]
        finiteDifferences (\x -> T.matmul x (AG.constant other)) vector
        finiteDifferences (T.matmul (AG.constant vector)) other
        finiteDifferences (\x -> T.matmul x (AG.constant right)) vector
        finiteDifferences (T.matmul (AG.constant vector)) right
        finiteDifferences (\x -> T.matmul x (AG.constant vector)) matrix
        finiteDifferences (T.matmul (AG.constant matrix)) vector
        finiteDifferences (\x -> T.matmul x (AG.constant batches)) vector
        finiteDifferences (T.matmul (AG.constant vector)) batches,
      testProperty "broadcast matmul pullbacks agree with finite differences" $ H.withTests 1 $ H.property $ do
        let left = T.fromList @'[2, 1, 1, 2] [0.1, 0.2, 0.3, 0.4 :: Double]
            right = T.fromList @'[3, 2, 1] [0.2, 0.3 .. 0.7 :: Double]
        finiteDifferences (\x -> T.matmul x (AG.constant right)) left
        finiteDifferences (T.matmul (AG.constant left)) right,
      testProperty "broadcast matmul pullbacks respect nonuniform output seeds" $ H.withTests 1 $ H.property $ do
        let left = T.fromList @'[2, 1, 1, 2] [1, 2, 3, 4 :: Double]
            right = T.fromList @'[3, 2, 1] [5, 6, 7, 8, 9, 10 :: Double]
            (_, backward) = AG.pullback (uncurry T.matmul) (left, right)
            (dl, dr) = backward (T.fromList @'[2, 3, 1, 1] [1, 2, 3, 4, 5, 6])
        values dl H.=== [46, 52, 109, 124]
        values dr H.=== [13, 18, 17, 24, 21, 30],
      testProperty "shared matrix gradients sum over multiple broadcast axes" $ H.withTests 1 $ H.property $ do
        let left = T.ones @'[2, 3, 1, 2] @Double
            right = T.ones @'[2, 1] @Double
            (_, (dl, dr)) = AG.backprop (uncurry T.matmul) (left, right)
        values dl H.=== replicate 12 1
        values dr H.=== [6, 6],
      testProperty "vector matmul accumulates both paths through a shared input" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [1, 2, 3 :: Double]
            (value, gradient) = differentiate (\x -> T.matmul x x) input
        values value H.=== [14]
        values gradient H.=== [2, 4, 6],
      testProperty "empty broadcast batches return zeros for nonempty shared operands" $ H.withTests 1 $ H.property $ do
        let left = T.ones @'[2, 3] @Double
            right = T.zeros @'[0, 3, 4] @Double
            (_, (dl, dr)) = AG.backprop (uncurry T.matmul) (left, right)
            emptyLeft = T.zeros @'[0, 2, 3] @Double
            sharedRight = T.ones @'[1, 3, 4] @Double
            (_, (emptyDL, sharedDR)) = AG.backprop (uncurry T.matmul) (emptyLeft, sharedRight)
        values dl H.=== replicate 6 0
        values dr H.=== []
        values emptyDL H.=== []
        values sharedDR H.=== replicate 12 0,
      testProperty "empty vector matmul produces a zero scalar and empty gradients" $ H.withTests 1 $ H.property $ do
        let input = T.zeros @'[0] @Double
            (value, gradient) = differentiate (\x -> T.matmul x x) input
        values value H.=== [0]
        values gradient H.=== [],
      testProperty "prepared broadcast matmul gradients accept changing vectors" $ H.withTests 1 $ H.property $ do
        let execute = T.run2 T.CPU $ \lhs rhs ->
              let (output, (dl, dr)) = AG.backprop (uncurry T.matmul) (lhs, rhs)
               in (output, dl, dr)
            vector = T.fromList @'[3] [1, 2, 3 :: Double]
            right = T.run T.CPU (T.ones @'[2, 3, 2] @Double)
            (first, firstDL, firstDR) = execute (T.run T.CPU vector) right
            (second, secondDL, secondDR) = execute (T.run T.CPU (2 * vector)) right
        T.toList first H.=== replicate 4 6
        T.toList firstDL H.=== [4, 4, 4]
        T.toList firstDR H.=== [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3]
        T.toList second H.=== replicate 4 12
        T.toList secondDL H.=== [4, 4, 4]
        T.toList secondDR H.=== [2, 2, 4, 4, 6, 6, 2, 2, 4, 4, 6, 6],
      testProperty "zero-size matrix contractions return zero gradients" $ H.withTests 1 $ H.property $ do
        let left = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            right = T.unsafeFromList [] :: T.Tensor '[3, 0] Double
            (_, gradient) = differentiate (\x -> T.matmul x (AG.constant right)) left
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
