{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Backend (tests) where

import qualified Data.Array.Accelerate as A
import qualified Hedgehog as H
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Type as Internal
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "backend"
    [ testProperty "CPU dispatch agrees with runCPU" (H.withTests 1 propCPUDispatch),
      testProperty "Accelerate representations can be unpacked and reused" (H.withTests 1 propAccelerateAccess)
    ]

propCPUDispatch :: H.Property
propCPUDispatch = H.property $ do
  let tensor = T.arange @4 (1 :: Int) 2
  T.toList (T.run T.CPU tensor) H.=== [1, 3, 5, 7]
  T.toList (T.run T.CPU tensor) H.=== T.toList (T.runCPU tensor)

propAccelerateAccess :: H.Property
propAccelerateAccess = H.property $ do
  let source = T.arange @3 (1 :: Int) 1
      modified =
        Internal.Tensor (A.map (+ 10) (Internal.unTensor source)) ::
          T.Tensor '[3] Int
      result = T.runCPU modified
      reused =
        T.fromData result ::
          T.Tensor '[3] Int
      element = T.fromGen @'[3] $ \i ->
        Internal.TensorElement (Internal.unTensorElement i * 2)
  A.toList (Internal.unTensorData result) H.=== [11, 12, 13]
  T.toList (T.runCPU reused) H.=== [11, 12, 13]
  T.toList (T.runCPU element) H.=== [0, 2, 4]
