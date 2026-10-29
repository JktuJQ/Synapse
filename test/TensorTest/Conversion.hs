{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module TensorTest.Conversion (tests) where

import Control.Exception (ErrorCall, displayException, evaluate, try)
import Control.Monad (void)
import Data.Int (Int32, Int64)
import Data.List (isInfixOf)
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
    [ testProperty "length validation stops at the first excess element" (H.withTests 1 propBoundedLength),
      testProperty "fromListData round-trips without a backend" propFromListDataRoundTrip,
      testProperty "fromListData supports scalars, empty tensors, and staged conversion" (H.withTests 1 propFromListDataEdges),
      testProperty "fromListData rejects mismatched lengths including empty shapes" (H.withTests 1 propFromListDataSize),
      testProperty "fromList round-trips through CPU backend" propFromListRoundTrip,
      testProperty "fromList rejects excess elements" (H.withTests 1 propFromListExcess),
      testProperty "fromList supports scalars and empty tensors" (H.withTests 1 propFromListEdges),
      testProperty "fromList rejects invalid element count" (H.withTests 1 propFromListRejectsInvalidSize),
      testProperty "supported dtypes round-trip" (H.withTests 1 propDTypes)
    ]

propBoundedLength :: H.Property
propBoundedLength = H.property $ do
  let values = 1 : 2 : 3 : error "unexpected traversal" :: [Int]
      rejects action = do
        result <- H.evalIO $ try @ErrorCall action
        case result of
          Left err -> H.assert ("at least 3" `isInfixOf` displayException err)
          Right _ -> H.failure
  rejects (void (evaluate (T.fromListData @'[2] values)))
  rejects (void (evaluate (T.fromList @'[2] values :: T.Tensor '[2] Int)))

propFromListDataRoundTrip :: H.Property
propFromListDataRoundTrip = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.float (Range.linearFracFrom 0 (-100) 100))
  T.toList (T.fromListData @'[2, 3] xs :: T.TensorData '[2, 3] T.Float32) H.=== xs

propFromListDataEdges :: H.Property
propFromListDataEdges = H.property $ do
  T.toList (T.fromListData @'[] [True]) H.=== [True]
  T.toList (T.fromListData @'[2, 0, 3] ([] :: [Int])) H.=== []
  T.toList (T.fromListData @'[2] [0, maxBound :: Word64]) H.=== [0, maxBound]
  let materialized = T.fromListData @'[2] [1.5, -2 :: Double]
  T.toList (T.run T.CPU (T.fromData materialized + 1)) H.=== [2.5, -1]

propFromListDataSize :: H.Property
propFromListDataSize = H.property $ do
  let rejects value = do
        result <- H.evalIO $ try @ErrorCall (evaluate value)
        case result of
          Left _ -> H.success
          Right _ -> H.failure
  mapM_ (rejects . T.fromListData @'[2]) [[], [1 :: Int], [1, 2, 3]]
  rejects (T.fromListData @'[0] [1 :: Int])

propFromListRoundTrip :: H.Property
propFromListRoundTrip = H.property $ do
  xs <- H.forAll $ Gen.list (Range.singleton 6) (Gen.float (Range.linearFracFrom 0 (-100) 100))
  let tensor = T.fromList @'[2, 3] xs :: T.Tensor '[2, 3] T.Float32
  T.toList (T.run T.CPU tensor) H.=== xs

propFromListExcess :: H.Property
propFromListExcess = H.property $ do
  result <- H.evalIO $ try @ErrorCall $ evaluate (T.fromList @'[2] [1, 2, 3] :: T.Tensor '[2] Int)
  case result of
    Left _ -> H.success
    Right _ -> H.failure

propFromListEdges :: H.Property
propFromListEdges = H.property $ do
  T.toList (T.run T.CPU (T.fromList @'[] [True])) H.=== [True]
  T.toList (T.run T.CPU (T.fromList @'[2, 0, 3] [] :: T.Tensor '[2, 0, 3] Int)) H.=== []

propFromListRejectsInvalidSize :: H.Property
propFromListRejectsInvalidSize = H.property $ do
  result <- H.evalIO $ try @ErrorCall $ evaluate (T.fromList @'[2, 3] [1.0, 2.0] :: T.Tensor '[2, 3] T.Float32)
  case result of
    Left _ -> H.success
    Right _ -> H.failure

propDTypes :: H.Property
propDTypes = H.property $ do
  T.toList (T.run T.CPU (T.fromList @'[2] [-3, 7 :: Int32])) H.=== [-3, 7]
  T.toList (T.run T.CPU (T.fromList @'[2] [-3, 7 :: Int64])) H.=== [-3, 7]
  T.toList (T.run T.CPU (T.fromList @'[2] [0, maxBound :: Word32])) H.=== [0, maxBound]
  T.toList (T.run T.CPU (T.fromList @'[2] [0, maxBound :: Word64])) H.=== [0, maxBound]
  T.toList (T.run T.CPU (T.fromList @'[2] [-0.5, 1.25 :: T.Float32])) H.=== [-0.5, 1.25]
  T.toList (T.run T.CPU (T.fromList @'[2] [-0.5, 1.25 :: T.Float64])) H.=== [-0.5, 1.25]
  T.toList (T.run T.CPU (T.fromList @'[2] [False, True])) H.=== [False, True]
