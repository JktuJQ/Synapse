{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module AutogradTest.Execution (tests) where

import qualified AutogradTest.Execution.Public
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Parameters t = Parameters {weights :: t '[2, 3] Double, bias :: t '[3] Double} deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor) = Parameters t

newtype Wrapped t = Wrapped (Parameters t) deriving (Generic)

type instance T.WithTensor t (Wrapped T.Tensor) = Wrapped t

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "execution"
    [ testProperty "nested records preserve outputs and independently registered gradients" $ H.withTests 1 $ H.property $ do
        let input = Wrapped (Parameters (T.fromList @'[2, 3] [1 .. 6]) (T.fromList @'[3] [7, 8, 9]))
            objective (Wrapped (Parameters w b)) = T.sum (w * w + T.broadcast b)
            (output, Wrapped (Parameters dw db)) = AG.backprop objective input
        values output H.=== [139]
        values dw H.=== [2, 4, 6, 8, 10, 12]
        values db H.=== [2, 2, 2],
      testProperty "frozen tensor features leave the head trainable" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            features = T.matmul input (T.fromList @'[2, 2] [2, 0, 0, 3 :: Double])
            headWeights = T.fromList @'[2, 1] [1, 2 :: Double]
            (value, gradient) = AG.backprop (T.sum . T.matmul (AG.constant features)) headWeights
        values value H.=== [44]
        values gradient H.=== [8, 18],
      testProperty "constant weights preserve input gradients" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] [1, 2, 3, 4 :: Double]
            frozen = T.fromList @'[2, 2] [2, 0, 0, 3 :: Double]
        values (AG.gradient (\x -> T.sum (T.matmul x (AG.constant frozen))) input) H.=== [2, 3, 2, 3],
      testProperty "pullbacks accept nonuniform seeds repeatedly" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [1, 2, 3 :: Double]
            (value, backward) = AG.pullback (\x -> x * x) input
        values value H.=== [1, 4, 9]
        values (backward (T.fromList @'[3] [2, 3, 5])) H.=== [4, 12, 30]
        values (backward 1) H.=== [2, 4, 6],
      testProperty "heterogeneous tuple outputs preserve shapes and element types" $ H.withTests 1 $ H.property $ do
        let input = (T.full @'[] (3 :: Double), T.full @'[2] (4 :: Float))
            (value, backward) = AG.pullback @_ @(T.Tensor '[] Double, T.Tensor '[2] Float) (\(x, y) -> (x * x, y * y)) input
            (dx, dy) = backward (2, 3)
        values (fst value) H.=== [9]
        T.toList (T.run T.CPU (snd value)) H.=== [16, 16]
        values dx H.=== [12]
        T.toList (T.run T.CPU dy) H.=== [24, 24],
      testProperty "unused tensor inputs receive shape-correct zeros" $ H.withTests 1 $ H.property $ do
        let input = (T.ones @'[2] @Double, T.ones @'[2, 3] @Double)
            (_, unused) = AG.gradient (T.sum . fst) input
        values unused H.=== replicate 6 0,
      testProperty "stopped history cuts only the selected path" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [1, 2, 3 :: Double]
            result = AG.gradient (\x -> T.sum (x * AG.stopGrad x)) input
        values result H.=== [1, 2, 3],
      testProperty "stopping a nested record cuts every field" $ H.withTests 1 $ H.property $ do
        let input = Wrapped (Parameters 2 3)
            objective p = let Wrapped (Parameters w b) = AG.stopGrad p in T.sum w + T.sum b
            Wrapped (Parameters dw db) = AG.gradient objective input
        values dw H.=== replicate 6 0
        values db H.=== replicate 3 0,
      testProperty "stopping singular expressions avoids unavailable derivatives" $ H.withTests 1 $ H.property $ do
        let function :: (AG.Context s) => AG.Tensor s '[] Double -> AG.Tensor s '[] Double
            function x = AG.stopGrad (sqrt x) + x
        values (AG.gradient function (T.zeros @'[] @Double)) H.=== [1],
      testProperty "Execution works without importing tensor constructors" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3] [1, 2, 3 :: Double]
        values (AutogradTest.Execution.Public.squaredGradient input) H.=== [2, 4, 6]
        values (AutogradTest.Execution.Public.constantGradient input input) H.=== [1, 2, 3],
      testProperty "autograd runs inside a prepared function" $ H.withTests 1 $ H.property $ do
        let function :: T.Tensor '[3] Double -> (T.Tensor '[] Double, T.Tensor '[3] Double)
            function = AG.backprop (\x -> T.sum (x * x))
            execute = T.run1 T.CPU function
            (first, df) = execute (T.run T.CPU (T.fromList @'[3] [1, 2, 3 :: Double]))
            (second, ds) = execute (T.run T.CPU (T.fromList @'[3] [4, 5, 6 :: Double]))
        T.toList first H.=== [14]
        T.toList df H.=== [2, 4, 6]
        T.toList second H.=== [77]
        T.toList ds H.=== [8, 10, 12]
    ]
