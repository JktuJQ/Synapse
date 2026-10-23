{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeApplications #-}

module AutogradTest.Tensor.Elementwise (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "elementwise gradients"
    [ testProperty "ReLU selects gradients and uses zero at the boundary" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[4] [-2, 0, 1, 3 :: Double]
            (value, gradient) = AG.backprop (\x -> T.select (x T.> 0) x 0) input
        values value H.=== [0, 0, 1, 3]
        values gradient H.=== [0, 0, 1, 1],
      testProperty "comparisons and boolean masks do not add gradient paths" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[4] [-2, 0, 1, 3 :: Double]
            function x = T.select (((x T.>= 0) T.&& (x T.<= 1)) T.|| ((x T.== 3) T.&& T.not (x T./= 3))) x (negate x)
            (value, gradient) = AG.backprop function input
        values value H.=== [2, 0, 1, 3]
        values gradient H.=== [-1, 1, 1, 1],
      testProperty "select routes nonuniform output seeds to both inputs" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[3] [1, 2, 3 :: Double]
            y = T.fromList @'[3] [4, 5, 6 :: Double]
            mask = T.fromList @'[3] [True, False, True]
            (value, backward) = AG.pullback (\(a, b) -> T.select mask a b) (x, y)
            (dx, dy) = backward (T.fromList @'[3] [2, 3, 5])
        values value H.=== [1, 5, 3]
        values dx H.=== [2, 0, 5]
        values dy H.=== [0, 3, 0],
      testProperty "both selected branches accumulate into a shared input" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [-2, 0, 3 :: Double]
            function x = T.select (x T.< 0) (x * x) (x + x)
        values (AG.gradient function input) H.=== [-4, 2, 2],
      testProperty "masked broadcast gradients sum only selected positions" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 3] [-1, 2, 3, 4, -5, 6 :: Double]
            bias = T.fromList @'[3] [1, 2, 3 :: Double]
            function (x, b) = T.sum (T.select (x T.> 0) (x + T.broadcast b) 0)
            (_, (dx, db)) = AG.backprop function (input, bias)
        values dx H.=== [0, 1, 1, 1, 0, 1]
        values db H.=== [1, 1, 2],
      testProperty "cond follows a scalar predicate derived from the input" $ H.withTests 1 $ H.property $ do
        let function x = T.cond (T.sum x T.> 0) (x * x) (negate x)
        values (AG.gradient function (T.fromList @'[2] [1, 2 :: Double])) H.=== [2, 4]
        values (AG.gradient function (T.fromList @'[2] [-3, 1 :: Double])) H.=== [-1, -1],
      testProperty "inactive selection gradients stay zero with infinite seeds" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [1, 2 :: Double]
            mask = T.fromList @'[2] [True, False]
            (_, backward) = AG.pullback (\(x, y) -> T.select mask x y) (input, input)
            (dx, dy) = backward (T.full @'[2] (1 / 0))
        values dx H.=== [1 / 0, 0]
        values dy H.=== [0, 1 / 0],
      testProperty "empty and scalar tensors preserve selection gradients" $ H.withTests 1 $ H.property $ do
        let empty = T.zeros @'[2, 0] @Double
        values (AG.gradient (\x -> T.select (x T.> 0) x 0) empty) H.=== []
        values (AG.gradient (\x -> T.select (x T.> 0) x 0) (T.full @'[] (2 :: Double))) H.=== [1],
      testProperty "element extrema divide gradients at ties" $ H.withTests 1 $ H.property $ do
        let x = T.fromList @'[3] [1, 2, 4 :: Double]
            y = T.fromList @'[3] [3, 2, 0 :: Double]
            (largest, (maxX, maxY)) = AG.backprop (\(a, b) -> T.max a b) (x, y)
            (smallest, (minX, minY)) = AG.backprop (\(a, b) -> T.min a b) (x, y)
        values largest H.=== [3, 2, 4]
        values smallest H.=== [1, 2, 0]
        values maxX H.=== [0, 0.5, 1]
        values maxY H.=== [1, 0.5, 0]
        values minX H.=== [1, 0.5, 0]
        values minY H.=== [0, 0.5, 1],
      testProperty "floating predicates produce usable masks without propagating gradients" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[4] [0 / 0, 1 / 0, -(1 / 0), 3 :: Double]
            function x = T.select (T.isNaN x T.|| T.isInfinite x) 0 x
            (value, gradient) = AG.backprop function input
        values value H.=== [0, 0, 0, 3]
        values gradient H.=== [0, 0, 0, 1],
      testProperty "index arithmetic and scalar construction preserve the tensor gradient" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [1, 2, 3 :: Double]
            function x = let element = T.index @'[1] x in T.scalar (element * element + sin element)
            (value, gradient) = AG.backprop function input
        approxList (values value) [4 + sin 2]
        approxList (values gradient) [0, 4 + cos 2, 0],
      testProperty "scalar indexing and construction can round-trip with gradients" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (3 :: Double)
            function x = T.scalar (T.index @'[] x * T.index @'[] x)
        values (AG.gradient function input) H.=== [6],
      testProperty "element inputs support selection and tensor-valued outputs" $ H.withTests 1 $ H.property $ do
        let function x = T.scalar (T.select (x T.> 0) (x * x) 0)
            (value, gradient) = AG.backprop function (T.constant (3 :: Double))
        values value H.=== [9]
        values (T.scalar gradient) H.=== [6]
        values (T.scalar (AG.gradient function (T.constant (-2 :: Double)))) H.=== [0],
      testProperty "element constants, extrema and scalar conditions preserve derivatives" $ H.withTests 1 $ H.property $ do
        let function x = T.cond (x T.< AG.constant (T.constant (0 :: Double))) (T.min x (-1)) (T.max x 1)
        values (T.scalar (AG.gradient function (T.constant (1 :: Double)))) H.=== [0.5]
        values (T.scalar (AG.gradient function (T.constant (-2 :: Double)))) H.=== [1],
      testProperty "prepared functions reuse a mask-dependent backward graph" $ H.withTests 1 $ H.property $ do
        let execute = T.run1 T.CPU (AG.backprop (\x -> T.select (x T.> 0) x 0))
            (firstValue, firstGradient) = execute (T.run T.CPU (T.fromList @'[3] [-2, 0, 3 :: Double]))
            (secondValue, secondGradient) = execute (T.run T.CPU (T.fromList @'[3] [2, -1, 0 :: Double]))
        T.toList firstValue H.=== [0, 0, 3]
        T.toList firstGradient H.=== [0, 0, 1]
        T.toList secondValue H.=== [2, 0, 0]
        T.toList secondGradient H.=== [1, 0, 0]
    ]
