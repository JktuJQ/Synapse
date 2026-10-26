{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

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

type instance T.WithTensor t (Inputs T.Tensor) = Inputs t

newtype Nested t = Nested {layer :: NN.Dense 1 1 t Double} deriving (Generic)

type instance T.WithTensor t (Nested T.Tensor) = Nested t

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
    [ testProperty "materialization preserves records with Bool and numeric tensors" $ H.withTests 1 $ H.property $ do
        let values :: Inputs T.Tensor
            values = Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False])
            result = T.run T.CPU values
        T.toList (numbers result) H.=== [2, 3]
        T.toList (mask result) H.=== [True, False],
      testProperty "prepared records accept changing tensors and masks" $ H.withTests 1 $ H.property $ do
        let function :: Inputs T.Tensor -> T.Tensor '[2] Double
            function p = T.select (mask p) (numbers p) 0
            execute = T.run1 T.CPU function
            first = T.run T.CPU (Inputs (T.fromList @'[2] [2, 3]) (T.fromList @'[2] [True, False]))
            second = T.run T.CPU (Inputs (T.fromList @'[2] [5, 7]) (T.fromList @'[2] [False, True]))
        T.toList (execute first) H.=== [2, 0]
        T.toList (execute second) H.=== [0, 7],
      testProperty "one prepared graph computes gradients and updates nested parameters" $ H.withTests 1 $ H.property $ do
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
