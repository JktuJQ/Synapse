{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE NoDeepSubsumption #-}

module TensorTest.Backend.Structure (tests) where

import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Inputs t = Inputs {numbers :: t '[2] Double, mask :: t '[2] Bool} deriving (Generic)

instance T.TensorStructure (Inputs T.Tensor) where
    type WithTensor t (Inputs T.Tensor) = Inputs t

newtype Nested t = Nested {layer :: NN.Dense 1 1 t Double} deriving (Generic)

instance T.TensorStructure (Nested T.Tensor) where
    type WithTensor t (Nested T.Tensor) = Nested t

class KeepFirst a where
    keepFirst :: a -> a -> a

instance KeepFirst (T.Tensor sh a) where
    keepFirst x _ = x

newtype WrappedTensor sh a = WrappedTensor (T.Tensor sh a)

step :: Nested T.Tensor -> T.Tensor '[] Double -> (T.Tensor '[] Double, Nested T.Tensor)
step parameters lr =
    let objective p = T.sum (NN.dense (layer p) (AG.constant (T.fromList @'[1] [2 :: Double])) ** 2)
        (loss, Nested gradients) = AG.backprop objective parameters
        NN.Dense w b = layer parameters
        updated =
            NN.Dense
                (w - T.broadcast @'[1, 1] lr * NN.denseWeights gradients)
                (b - T.broadcast @'[1] lr * NN.denseBias gradients)
     in (loss, Nested updated)

coefficients :: Nested T.TensorData -> [Double]
coefficients (Nested p) = T.toList (NN.denseWeights p) ++ T.toList (NN.denseBias p)

tests :: TestTree
tests =
    testGroup
        "structured execution"
        [ testProperty "flatten and rebuild preserve a nested layer" $ H.withTests 1 $ H.property $ do
            let original = Nested (NN.Dense (T.fromList @'[1, 1] [3]) (T.fromList @'[1] [2]))
                (weights, bias) = T.flatten @(Nested T.Tensor) @T.Tensor original
                restored = T.rebuild @(Nested T.Tensor) @T.Tensor (weights, bias)
            T.toList (T.run T.CPU weights) H.=== [3]
            T.toList (T.run T.CPU bias) H.=== [2]
            coefficients (T.run T.CPU restored) H.=== [3, 2]
        , testProperty "materialized products can be edited and rebuilt" $ H.withTests 1 $ H.property $ do
            let original = Inputs (T.fromListData @'[2] [2, 3]) (T.fromListData @'[2] [True, False])
                (values, _) = T.flatten @(Inputs T.Tensor) @T.TensorData original
                replacement = T.fromListData @'[2] [False, True]
                restored = T.rebuild @(Inputs T.Tensor) @T.TensorData (values, replacement)
            T.toList (numbers restored) H.=== [2, 3]
            T.toList (mask restored) H.=== [False, True]
        , testProperty "map changes representation in a mixed tensor record" $ H.withTests 1 $ H.property $ do
            let input = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                Inputs (WrappedTensor numbers') (WrappedTensor mask') =
                    T.mapTensorStructure @(Inputs T.Tensor) @T.Tensor @WrappedTensor WrappedTensor input
            T.toList (T.run T.CPU numbers') H.=== [2, 3]
            T.toList (T.run T.CPU mask') H.=== [True, False]
        , testProperty "zip combines Bool tensors without differentiability" $ H.withTests 1 $ H.property $ do
            let left = T.fromList @'[2] [True, False]
                right = T.fromList @'[2] [False, True]
                result = T.zipWithTensorStructure @(T.Tensor '[2] Bool) @KeepFirst keepFirst left right
            T.toList (T.run T.CPU result) H.=== [True, False]
        , testProperty "materialization preserves records with Bool and numeric tensors" $ H.withTests 1 $ H.property $ do
            let values :: Inputs T.Tensor
                values = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
                result = T.run T.CPU values
            T.toList (numbers result) H.=== [2, 3]
            T.toList (mask result) H.=== [True, False]
        , testProperty "prepared records accept changing tensors and masks" $ H.withTests 1 $ H.property $ do
            let function :: Inputs T.Tensor -> T.Tensor '[2] Double
                function p = T.select (mask p) (numbers p) 0
                execute = T.run1 T.CPU function
                first = T.run T.CPU (Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False]))
                second = T.run T.CPU (Inputs (T.fromList @'[2] [5, 7]) (T.fromList @'[2] [False, True]))
            T.toList (execute first) H.=== [2, 0]
            T.toList (execute second) H.=== [0, 7]
        , testProperty "one prepared graph computes gradients and updates nested parameters" $ H.withTests 1 $ H.property $ do
            let initial :: Nested T.Tensor
                initial = Nested (NN.Dense 1 0)
                parameters = T.run T.CPU initial
                execute = T.run2 T.CPU step
                zero = T.run T.CPU (T.full @'[] (0 :: Double))
                rate = T.run T.CPU (T.full @'[] (0.1 :: Double))
                (initialLoss, unchanged) = execute parameters zero
                (_, first) = execute parameters rate
                (nextLoss, second) = execute first rate
            T.toList initialLoss H.=== [4]
            coefficients unchanged H.=== [1, 0]
            approxList (coefficients first) [0.2, -0.4]
            approxList (T.toList nextLoss) [0]
            approxList (coefficients second) [0.2, -0.4]
        ]
