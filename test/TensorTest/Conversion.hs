{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Conversion (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import Data.Int (Int32, Int64)
import Data.Word (Word32, Word64)
import qualified Hedgehog as H
import qualified Hedgehog.Gen as Gen
import qualified Hedgehog.Range as Range
import qualified Synapse.Tensor as T
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

tests :: TestTree
tests =
  testGroup
    "conversion"
    [ testProperty "fromList round-trips through CPU backend" propFromListRoundTrip,
      testProperty "fromList rejects excess elements" (H.withTests 1 propFromListExcess),
      testProperty "fromList supports scalars and empty tensors" (H.withTests 1 propFromListEdges),
      testProperty "fromList rejects invalid element count" (H.withTests 1 propFromListRejectsInvalidSize),
      testProperty "supported dtypes round-trip" (H.withTests 1 propDTypes)
    ]

propFromListRoundTrip :: H.Property
propFromListRoundTrip = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.float (Range.linearFracFrom 0 (-100) 100))
  let tensor = T.fromList @'[2, 3] xs :: T.Tensor '[2, 3] T.Float32
  T.toList (T.runCPU tensor) H.=== xs

propFromListExcess :: H.Property
propFromListExcess = H.property $ do
  result <- H.evalIO $ try @ErrorCall $ evaluate $ T.fromList @'[2] [1, 2, 3 :: Int]
  case result of
    Left _ -> H.success
    Right _ -> H.failure

propFromListEdges :: H.Property
propFromListEdges = H.property $ do
  T.toList (T.runCPU (T.fromList @'[] [True])) H.=== [True]
  T.toList (T.runCPU (T.fromList @'[2, 0, 3] [] :: T.Tensor '[2, 0, 3] Int)) H.=== []

propFromListRejectsInvalidSize :: H.Property
propFromListRejectsInvalidSize = H.property $ do
  result <- H.evalIO $ try @ErrorCall $ evaluate $ T.fromList @'[2, 3] [1.0, 2.0 :: T.Float32]
  case result of
    Left _ -> H.success
    Right _ -> H.failure

propDTypes :: H.Property
propDTypes = H.property $ do
  T.toList (T.runCPU (T.fromList @'[2] [-3, 7 :: Int32])) H.=== [-3, 7]
  T.toList (T.runCPU (T.fromList @'[2] [-3, 7 :: Int64])) H.=== [-3, 7]
  T.toList (T.runCPU (T.fromList @'[2] [0, maxBound :: Word32])) H.=== [0, maxBound]
  T.toList (T.runCPU (T.fromList @'[2] [0, maxBound :: Word64])) H.=== [0, maxBound]
  T.toList (T.runCPU (T.fromList @'[2] [-0.5, 1.25 :: T.Float32])) H.=== [-0.5, 1.25]
  T.toList (T.runCPU (T.fromList @'[2] [-0.5, 1.25 :: T.Float64])) H.=== [-0.5, 1.25]
  T.toList (T.runCPU (T.fromList @'[2] [False, True])) H.=== [False, True]
