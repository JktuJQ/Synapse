{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Weights.Safetensors (tests) where

import qualified Data.ByteString.Builder as B
import qualified Data.ByteString.Lazy as LBS
import qualified Data.ByteString.Lazy.Char8 as LBS8
import Data.Int (Int32, Int64)
import Data.List (isInfixOf)
import Data.Typeable (Typeable)
import Data.Word (Word32, Word64)
import GHC.Float (castDoubleToWord64, castFloatToWord32)
import qualified Hedgehog as H
import qualified Synapse.NN as NN
import qualified Synapse.NN.Weights as W
import qualified Synapse.NN.Weights.Safetensors as S
import qualified Synapse.Tensor as T
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

fixture :: String -> B.Builder -> LBS.ByteString
fixture header body = B.toLazyByteString $ B.word64LE (fromIntegral (length header)) <> B.lazyByteString (LBS8.pack header) <> body

entry :: String -> String -> String -> String -> String
entry name dtype dimensions offsets = show name ++ ":{\"dtype\":" ++ show dtype ++ ",\"shape\":" ++ dimensions ++ ",\"data_offsets\":" ++ offsets ++ "}"

roundTrip :: W.Weights -> H.PropertyT IO W.Weights
roundTrip weights = do
  bytes <- H.evalEither (S.encodeSafetensors weights)
  H.evalEither (S.decodeSafetensors bytes)

elements :: forall a. (Typeable a) => W.FieldPath -> W.Weights -> H.PropertyT IO [a]
elements path weights = do
  tensor <- H.evalMaybe (W.lookupWeight path weights)
  values <- H.evalMaybe (W.weightTensorValues @a tensor)
  pure (W.valuesToList values)

reject :: LBS.ByteString -> H.PropertyT IO ()
reject bytes = case S.decodeSafetensors bytes of
  Left _ -> pure ()
  Right _ -> H.failure

tests :: TestTree
tests =
  testGroup
    "safetensors"
    [ testProperty "Dense parameters round trip into their original structure" $ H.withTests 1 $ H.property $ do
        let source = NN.Dense (T.fromListData @'[2, 1] [1, 2 :: Double]) (T.fromListData @'[1] [3])
        weights <- roundTrip (NN.extractWeights source)
        restored <- H.evalEither (NN.loadWeights weights source)
        T.toList (NN.denseWeights restored) H.=== [1, 2]
        T.toList (NN.denseBias restored) H.=== [3],
      testProperty "all supported scalar dtypes retain their Haskell types" $ H.withTests 1 $ H.property $ do
        let source =
              NN.extractWeights
                ( T.fromListData @'[2] [minBound, maxBound :: Int],
                  T.fromListData @'[2] [minBound, maxBound :: Int32],
                  T.fromListData @'[2] [minBound, maxBound :: Int64],
                  T.fromListData @'[2] [0, maxBound :: Word32],
                  T.fromListData @'[2] [0, maxBound :: Word64],
                  T.fromListData @'[2] [False, True],
                  T.fromListData @'[2] [1.25, -2.5 :: Float],
                  T.fromListData @'[2] [1e200, -1e-200 :: Double]
                )
        weights <- roundTrip source
        elements @Int ["0"] weights >>= (H.=== [minBound, maxBound])
        elements @Int32 ["1"] weights >>= (H.=== [minBound, maxBound])
        elements @Int64 ["2"] weights >>= (H.=== [minBound, maxBound])
        elements @Word32 ["3"] weights >>= (H.=== [0, maxBound])
        elements @Word64 ["4"] weights >>= (H.=== [0, maxBound])
        elements @Bool ["5"] weights >>= (H.=== [False, True])
        elements @Float ["6"] weights >>= (H.=== [1.25, -2.5])
        elements @Double ["7"] weights >>= (H.=== [1e200, -1e-200]),
      testProperty "NaN infinities and signed zero preserve floating point bits" $ H.withTests 1 $ H.property $ do
        let doubles = [0, -0, 0 / 0, 1 / 0, -(1 / 0) :: Double]
            floats = [0, -0, 0 / 0, 1 / 0, -(1 / 0) :: Float]
        weights <- roundTrip $ NN.extractWeights (T.fromListData @'[5] doubles, T.fromListData @'[5] floats)
        actualD <- elements @Double ["0"] weights
        actualF <- elements @Float ["1"] weights
        map castDoubleToWord64 actualD H.=== map castDoubleToWord64 doubles
        map castFloatToWord32 actualF H.=== map castFloatToWord32 floats,
      testProperty "scalar empty and empty dictionary encodings round trip" $ H.withTests 1 $ H.property $ do
        weights <- roundTrip $ NN.extractWeights (T.fromListData @'[] [True], T.fromListData @'[2, 0] ([] :: [Float]))
        elements @Bool ["0"] weights >>= (H.=== [True])
        elements @Float ["1"] weights >>= (H.=== [])
        empty <- roundTrip (NN.extractWeights ())
        map fst (W.weightsToList empty) H.=== [],
      testProperty "arbitrary paths and Unicode names survive repeated encoding" $ H.withTests 1 $ H.property $ do
        let paths = [[], [""], ["a.b"], ["a", "b"], ["a/b", "~", ""], ["__metadata__"], ["веса", "\x1f600"]]
            tensor = W.weightTensor (T.fromListData @'[] [1 :: Double])
        source <- H.evalEither $ W.weightsFromList $ map (,tensor) paths
        weights <- roundTrip source >>= roundTrip
        map fst (W.weightsToList weights) H.=== map fst (W.weightsToList source),
      testProperty "valid escaped surrogate pairs decode to the same Unicode name" $ H.withTests 1 $ H.property $ do
        let header = "{\"\\uD83D\\uDE00\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]},\"__metadata__\":{\"note\":\"\\uD83D\\uDE00\"}}"
        weights <- H.evalEither $ S.decodeSafetensors $ fixture header (B.floatLE 2.5)
        elements @Float ["\x1f600"] weights >>= (H.=== [2.5])
        restored <- roundTrip weights
        elements @Float ["\x1f600"] restored >>= (H.=== [2.5]),
      testProperty "external dtype names are decoded without host Int conversion" $ H.withTests 1 $ H.property $ do
        let header = "{" ++ entry "layer.weight" "I64" "[2]" "[0,16]" ++ "}"
        weights <- H.evalEither $ S.decodeSafetensors $ fixture header (B.int64LE (-2) <> B.int64LE 9007199254740993)
        elements @Int64 ["layer", "weight"] weights >>= (H.=== [-2, 9007199254740993]),
      testProperty "external names containing slashes and tildes remain literal" $ H.withTests 1 $ H.property $ do
        let name = "/external~name.weight"
            header = "{" ++ entry name "F32" "[]" "[0,4]" ++ "}"
        weights <- H.evalEither $ S.decodeSafetensors $ fixture header (B.floatLE 1)
        elements @Float ["/external~name", "weight"] weights >>= (H.=== [1])
        bytes <- H.evalEither $ S.encodeSafetensors weights
        H.assert $ name `isInfixOf` LBS8.unpack (LBS.take 200 bytes)
        restored <- H.evalEither $ S.decodeSafetensors bytes
        elements @Float ["/external~name", "weight"] restored >>= (H.=== [1]),
      testProperty "header order and data order may differ" $ H.withTests 1 $ H.property $ do
        let header = "{" ++ entry "later" "F32" "[]" "[4,8]" ++ "," ++ entry "earlier" "F32" "[]" "[0,4]" ++ "}"
        weights <- H.evalEither $ S.decodeSafetensors $ fixture header (B.floatLE 10 <> B.floatLE 20)
        elements @Float ["earlier"] weights >>= (H.=== [10])
        elements @Float ["later"] weights >>= (H.=== [20]),
      testProperty "duplicate and Unicode-equivalent names are rejected" $ H.withTests 1 $ H.property $ do
        reject $ fixture ("{" ++ entry "x" "F32" "[0]" "[0,0]" ++ "," ++ entry "x" "F32" "[0]" "[0,0]" ++ "}") mempty
        reject $ fixture "{\"x\":{\"dtype\":\"F32\",\"dtype\":\"F64\",\"shape\":[0],\"data_offsets\":[0,0]}}" mempty,
      testProperty "unsupported dtypes and tuple elements report explicit errors" $ H.withTests 1 $ H.property $ do
        case S.decodeSafetensors $ fixture ("{" ++ entry "x" "F16" "[0]" "[0,0]" ++ "}") mempty of
          Left (S.UnsupportedDType "F16") -> pure ()
          _ -> H.failure
        case S.encodeSafetensors (NN.extractWeights (T.fromListData @'[] [(1 :: Double, True)])) of
          Left (S.UnsupportedElementType _) -> pure ()
          _ -> H.failure,
      testProperty "invalid shapes offsets lengths and boolean values are rejected" $ H.withTests 1 $ H.property $ do
        let invalid dtype shape offsets body = reject $ fixture ("{" ++ entry "x" dtype shape offsets ++ "}") body
        invalid "F32" "[-1]" "[0,0]" mempty
        invalid "F32" "[1.5]" "[0,4]" (B.floatLE 0)
        invalid "F32" "[2]" "[0,4]" (B.floatLE 0)
        invalid "F32" "[1]" "[0,8]" (B.floatLE 0)
        invalid "F32" "[1]" "[4,8]" (B.floatLE 0 <> B.floatLE 0)
        invalid "F32" "[1]" "[4,0]" (B.floatLE 0)
        invalid "F32" "[1]" "[0,4]" (B.floatLE 0 <> B.word8 1)
        invalid "BOOL" "[1]" "[0,1]" (B.word8 2)
        invalid "F32" "[9223372036854775808]" "[0,0]" mempty
        reject $ fixture ("{" ++ entry "a" "F32" "[1]" "[0,4]" ++ "," ++ entry "b" "F32" "[1]" "[0,4]" ++ "}") (B.floatLE 0),
      testProperty "malformed headers UTF-8 and unpaired surrogates are rejected" $ H.withTests 1 $ H.property $ do
        reject LBS.empty
        reject $ B.toLazyByteString $ B.word64LE 100000001
        reject $ B.toLazyByteString $ B.word64LE 10 <> B.word8 123
        reject $ fixture "[]" mempty
        reject $ fixture "{\"__metadata__\":{\"x\":1}}" mempty
        reject $ fixture "{\"__metadata__\":{\"x\":\"\\uD800\"}}" mempty
        reject $ B.toLazyByteString $ B.word64LE 4 <> B.byteString (LBS.toStrict $ LBS.pack [123, 255, 125, 32]),
      testProperty "host Int metadata is validated" $ H.withTests 1 $ H.property $ do
        reject $ fixture "{\"__metadata__\":{\"synapse.host_int\":\"[\\\"missing\\\"]\"}}" mempty
        reject $ fixture ("{" ++ entry "x" "F32" "[]" "[0,4]" ++ ",\"__metadata__\":{\"synapse.host_int\":\"[\\\"x\\\"]\"}}") (B.floatLE 1),
      testProperty "foreign Dense weights can be renamed and transposed before loading" $ H.withTests 1 $ H.property $ do
        let header = "{" ++ entry "hidden.weight" "F64" "[2,3]" "[0,48]" ++ "," ++ entry "hidden.bias" "F64" "[2]" "[48,64]" ++ "}"
        foreignWeights <- H.evalEither $ S.decodeSafetensors $ fixture header $ foldMap B.doubleLE [1, 2, 3, 4, 5, 6, 10, 20]
        renamed <- H.evalEither $ W.renameWeights (\path -> case path of ["hidden", "weight"] -> ["denseWeights"]; ["hidden", "bias"] -> ["denseBias"]; _ -> path) foreignWeights
        adapted <- H.evalEither $ W.traverseWeights (\path tensor -> if path == ["denseWeights"] then W.permuteWeightTensor [1, 0] tensor else Right tensor) renamed
        let template = NN.Dense (T.fromListData @'[3, 2] (replicate 6 (0 :: Double))) (T.fromListData @'[2] [0, 0])
        restored <- H.evalEither $ NN.loadWeights adapted template
        T.toList (NN.denseWeights restored) H.=== [1, 4, 2, 5, 3, 6]
        T.toList (NN.denseBias restored) H.=== [10, 20]
        bytes <- H.evalEither $ S.encodeSafetensors adapted
        H.assert $ "denseWeights" `isInfixOf` LBS8.unpack (LBS.take 200 bytes)
    ]
