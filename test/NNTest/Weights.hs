{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Weights (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import Control.Monad (void)
import Data.Proxy (Proxy (..))
import Data.Typeable (cast, typeRep)
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified NNTest.Weights.Public as Public
import qualified NNTest.Weights.Safetensors as Safetensors
import qualified Synapse.NN as NN
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.Tensor as T
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Parameters t a = Parameters
  { layer :: NN.Dense 2 1 t a,
    learningRate :: Double,
    labels :: [String],
    enabled :: Bool,
    threshold :: Maybe Double
  }
  deriving (Generic)

parameters :: Parameters T.TensorData Double
parameters = Parameters (NN.Dense (T.fromListData @'[2, 1] [0.25, 0.5]) (T.fromListData @'[1] [0.1])) 0.01 ["first", "second"] True (Just 0.5)

checkWeights :: (H.MonadTest m) => Parameters T.TensorData Double -> m ()
checkWeights restored = do
  T.toList (NN.denseWeights (layer restored)) H.=== [0.25, 0.5]
  T.toList (NN.denseBias (layer restored)) H.=== [0.1]

rejects :: (H.MonadTest m) => Weights.WeightsError -> Either Weights.WeightsError a -> m ()
rejects expected result = case result of
  Left actual -> actual H.=== expected
  Right _ -> H.failure

tests :: TestTree
tests =
  testGroup
    "weight state"
    [ Safetensors.tests,
      testProperty "dictionary adapters reject collisions and preserve selected entries" $ H.withTests 1 $ H.property $ do
        let tensor = Weights.weightTensor (T.fromListData @'[] [1 :: Double])
        dictionary <- H.evalEither $ Weights.weightsFromList [(["a"], tensor), (["b"], tensor)]
        rejects (Weights.DuplicateField ["same"]) $ Weights.renameWeights (const ["same"]) dictionary
        rejects (Weights.DuplicateField ["a"]) $ Weights.insertWeight ["a"] tensor dictionary
        renamed <- H.evalEither $ Weights.renameWeights ("layer" :) dictionary
        map fst (Weights.weightsToList renamed) H.=== [["layer", "a"], ["layer", "b"]]
        let selected = Weights.filterWeights (== ["layer", "a"]) renamed
        map fst (Weights.weightsToList selected) H.=== [["layer", "a"]]
        inserted <- H.evalEither $ Weights.insertWeight ["new"] tensor $ Weights.deleteWeight ["layer", "a"] selected
        map fst (Weights.weightsToList inserted) H.=== [["new"]],
      testProperty "host tensor adapters transpose reshape and split packed projections" $ H.withTests 1 $ H.property $ do
        source <- H.evalEither $ Weights.weightTensorFromList [3, 2] [1, 2, 3, 4, 5, 6 :: Double]
        transposed <- H.evalEither $ Weights.permuteWeightTensor [1, 0] source
        Weights.weightTensorShape transposed H.=== [2, 3]
        fmap Weights.valuesToList (Weights.weightTensorValues @Double transposed) H.=== Just [1, 3, 5, 2, 4, 6]
        reshaped <- H.evalEither $ Weights.reshapeWeightTensor [6] transposed
        fmap Weights.valuesToList (Weights.weightTensorValues @Double reshaped) H.=== Just [1, 3, 5, 2, 4, 6]
        middle <- H.evalEither $ Weights.sliceWeightTensor 0 1 1 source
        fmap Weights.valuesToList (Weights.weightTensorValues @Double middle) H.=== Just [3, 4]
        column <- H.evalEither $ Weights.sliceWeightTensor 1 1 1 source
        fmap Weights.valuesToList (Weights.weightTensorValues @Double column) H.=== Just [2, 4, 6]
        rejects (Weights.InvalidAxes [0, 0] [3, 2]) $ Weights.permuteWeightTensor [0, 0] source
        rejects (Weights.InvalidSlice 1 1 2 [3, 2]) $ Weights.sliceWeightTensor 1 1 2 source
        rejects (Weights.ElementCountMismatch 5 6) $ Weights.reshapeWeightTensor [5] source,
      testProperty "host adapters support scalar identity empty outputs and type inspection" $ H.withTests 1 $ H.property $ do
        scalar <- H.evalEither $ Weights.weightTensorFromList [] [True]
        same <- H.evalEither $ Weights.permuteWeightTensor [] scalar
        fmap Weights.valuesToList (Weights.weightTensorValues @Bool same) H.=== Just [True]
        fmap Weights.valuesToList (Weights.weightTensorValues @Double same) H.=== Nothing
        empty <- H.evalEither $ Weights.weightTensorFromList [2, 0] ([] :: [Double])
        transposed <- H.evalEither $ Weights.permuteWeightTensor [1, 0] empty
        Weights.weightTensorShape transposed H.=== [0, 2]
        fmap Weights.valuesToList (Weights.weightTensorValues @Double transposed) H.=== Just []
        rejects (Weights.InvalidSlice (-1) 0 0 []) $ Weights.sliceWeightTensor (-1) 0 0 scalar,
      testProperty "nested records extract only tensors and preserve destination metadata" $ H.withTests 1 $ H.property $ do
        let dictionary = NN.extractWeights parameters
            destination = Parameters (NN.Dense (T.fromListData @'[2, 1] [0, 0]) (T.fromListData @'[1] [0])) 0.3 ["destination"] False Nothing
        map fst (Weights.weightsToList dictionary) H.=== [["layer", "denseBias"], ["layer", "denseWeights"]]
        restored <- H.evalEither (NN.loadWeights dictionary destination)
        checkWeights restored
        learningRate restored H.=== 0.3
        labels restored H.=== ["destination"]
        enabled restored H.=== False
        threshold restored H.=== Nothing,
      testProperty "external codecs can reconstruct entries without static shapes" $ H.withTests 1 $ H.property $ do
        let encodeDecode (path, tensor) = Weights.withWeightTensor tensor $ \dimensions values -> (,) path <$> Weights.weightTensorFromFunction dimensions (Weights.valueAt values)
        entries <- H.evalEither (traverse encodeDecode (Weights.weightsToList (NN.extractWeights parameters)))
        dictionary <- H.evalEither (Weights.weightsFromList (reverse entries))
        restored <- H.evalEither (NN.loadWeights dictionary parameters)
        checkWeights restored,
      testProperty "entry inspection exposes shape, type, and typed row-major elements" $ H.withTests 1 $ H.property $ do
        let tensor = Weights.weightTensor (T.fromListData @'[2, 1] [1, 2 :: Double])
        Weights.weightTensorShape tensor H.=== [2, 1]
        Weights.weightTensorType tensor H.=== typeRep (Proxy @Double)
        Weights.withWeightTensor tensor (\_ values -> cast (Weights.valuesToList values) :: Maybe [Double]) H.=== Just [1, 2],
      testProperty "element views support indexed reads and explicit list conversion" $ H.withTests 1 $ H.property $ do
        let tensor = Weights.weightTensor (T.fromListData @'[2, 2] [10, 20, 30, 40 :: Int])
            inspect _ values =
              ( Weights.valuesSize values,
                cast [Weights.valueAt values 3, Weights.valueAt values 0] :: Maybe [Int],
                cast (Weights.valuesToList values) :: Maybe [Int]
              )
        Weights.withWeightTensor tensor inspect H.=== (4, Just [40, 10], Just [10, 20, 30, 40]),
      testProperty "indexed views reject negative, past-end and empty indices" $ H.withTests 1 $ H.property $ do
        let rejectsIndex source index = do
              result <- H.evalIO $ Weights.withWeightTensor source $ \_ values -> try @ErrorCall (void (evaluate (Weights.valueAt values index)))
              case result of
                Left _ -> pure ()
                Right _ -> H.failure
            tensor = Weights.weightTensor (T.fromListData @'[1] [1 :: Double])
        rejectsIndex tensor (-1)
        rejectsIndex tensor 1
        rejectsIndex (Weights.weightTensor (T.fromListData @'[0] ([] :: [Double]))) 0,
      testProperty "indexed construction handles row-major arrays, scalars and empty shapes" $ H.withTests 1 $ H.property $ do
        tensor <- H.evalEither (Weights.weightTensorFromFunction [2, 3] (\index -> fromIntegral (index * 2) :: Double))
        dictionary <- H.evalEither (Weights.weightsFromList [([], tensor)])
        restored <- H.evalEither (NN.loadWeights dictionary (T.fromListData @'[2, 3] (replicate 6 (0 :: Double))))
        T.toList restored H.=== [0, 2, 4, 6, 8, 10]
        scalar <- H.evalEither (Weights.weightTensorFromFunction [] (== 0))
        Weights.withWeightTensor scalar (\_ values -> (Weights.valuesSize values, cast (Weights.valueAt values 0) :: Maybe Bool)) H.=== (1, Just True)
        empty <- H.evalEither (Weights.weightTensorFromFunction [2, 0] (\_ -> error "empty source was read" :: Double))
        Weights.withWeightTensor empty (\_ values -> (Weights.valuesSize values, length (Weights.valuesToList values))) H.=== (0, 0)
        rejects (Weights.InvalidShape [-1]) (Weights.weightTensorFromFunction [-1] (\_ -> error "invalid source was read" :: Double)),
      testProperty "scalars, empty tensors, mixed element types and tuple paths round trip" $ H.withTests 1 $ H.property $ do
        let source = (T.fromListData @'[] [True], T.fromListData @'[0, 2] ([] :: [Float]), T.fromListData @'[2] [1, 2 :: Int])
            destination = (T.fromListData @'[] [False], T.fromListData @'[0, 2] ([] :: [Float]), T.fromListData @'[2] [0, 0 :: Int])
            dictionary = NN.extractWeights source
        map fst (Weights.weightsToList dictionary) H.=== [["0"], ["1"], ["2"]]
        (scalar, empty, integers) <- H.evalEither (NN.loadWeights dictionary destination)
        T.toList scalar H.=== [True]
        T.toList empty H.=== []
        T.toList integers H.=== [1, 2],
      testProperty "loading rejects missing and unexpected fields" $ H.withTests 1 $ H.property $ do
        let entries = Weights.weightsToList (NN.extractWeights parameters)
        incomplete <- H.evalEither (Weights.weightsFromList (drop 1 entries))
        rejects (Weights.MissingFields [["layer", "denseBias"]]) (NN.loadWeights incomplete parameters)
        extra <- H.evalEither (Weights.weightsFromList ((["extra"], Weights.weightTensor (T.fromListData @'[] [0 :: Double])) : entries))
        rejects (Weights.UnexpectedFields [["extra"]]) (NN.loadWeights extra parameters),
      testProperty "duplicate paths are rejected before weights can be overwritten" $ H.withTests 1 $ H.property $ do
        let entry = (["weight"], Weights.weightTensor (T.fromListData @'[1] [1 :: Double]))
        rejects (Weights.DuplicateField ["weight"]) (Weights.weightsFromList [entry, entry]),
      testProperty "loading rejects shapes rather than reshaping equal-sized arrays" $ H.withTests 1 $ H.property $ do
        let source = NN.extractWeights (T.fromListData @'[2] [1, 2 :: Double])
            destination = T.fromListData @'[1, 2] [0, 0 :: Double]
        rejects (Weights.ShapeMismatch [] [1, 2] [2]) (NN.loadWeights source destination),
      testProperty "loading rejects implicit element type conversion" $ H.withTests 1 $ H.property $ do
        let source = NN.extractWeights (T.fromListData @'[1] [1 :: Float])
            destination = T.fromListData @'[1] [0 :: Double]
        rejects (Weights.ElementTypeMismatch [] (typeRep (Proxy @Double)) (typeRep (Proxy @Float))) (NN.loadWeights source destination),
      testProperty "dynamic entries validate dimensions and element counts" $ H.withTests 1 $ H.property $ do
        rejects (Weights.InvalidShape [-1]) (Weights.weightTensorFromList [-1] ([] :: [Double]))
        rejects (Weights.InvalidShape [maxBound, 2]) (Weights.weightTensorFromList [maxBound, 2] ([] :: [Double]))
        rejects (Weights.ElementCountMismatch 1 0) (Weights.weightTensorFromList [] ([] :: [Double]))
        rejects (Weights.ElementCountMismatch 2 1) (Weights.weightTensorFromList [2] [1 :: Double])
        rejects (Weights.ElementCountMismatch 1 2) (Weights.weightTensorFromList [1] (1 : 2 : error "unexpected traversal" :: [Double]))
        scalar <- H.evalEither (Weights.weightTensorFromList [] [3 :: Double])
        restored <- H.evalEither (NN.loadWeights (NN.extractWeights (T.fromListData @'[] [3 :: Double])) (T.fromListData @'[] [0 :: Double]))
        Weights.weightTensorShape scalar H.=== []
        T.toList restored H.=== [3],
      testProperty "optional tensor fields preserve the destination constructor" $ H.withTests 1 $ H.property $ do
        let optional = Just (T.fromListData @'[1] [2 :: Double])
        rejects (Weights.UnexpectedFields [["Just"]]) (NN.loadWeights (NN.extractWeights optional) (Nothing :: Maybe (T.TensorData '[1] Double))),
      testProperty "text metadata is not evaluated during weight extraction" $ H.withTests 1 $ H.property $ do
        let source = (T.fromListData @'[] [1 :: Double], error "text metadata was inspected" :: String)
        map fst (Weights.weightsToList (NN.extractWeights source)) H.=== [["0"]],
      testProperty "public codecs and manual nested traversal round trip without internal imports" $ H.withTests 1 $ H.property $ do
        let source = Public.Manual (layer parameters) (T.fromListData @'[] [True]) (Public.Settings (+ 1))
            destination = Public.Manual (NN.Dense (T.fromListData @'[2, 1] [0, 0]) (T.fromListData @'[1] [0])) (T.fromListData @'[] [False]) (Public.Settings (* 2))
        encoded <- H.evalEither (Public.encode (NN.extractWeights source))
        decoded <- H.evalEither (Public.decode encoded)
        map fst (Weights.weightsToList decoded) H.=== [["enabled"], ["projection", "denseBias"], ["projection", "denseWeights"]]
        Public.Manual restored enabledTensor (Public.Settings configure) <- H.evalEither (NN.loadWeights decoded destination)
        T.toList (NN.denseWeights restored) H.=== [0.25, 0.5]
        T.toList (NN.denseBias restored) H.=== [0.1]
        T.toList enabledTensor H.=== [True]
        configure 3 H.=== 6,
      testProperty "Generic traversal accepts opaque configuration through its public instance" $ H.withTests 1 $ H.property $ do
        let source = Public.GenericParameters (layer parameters) (Public.Settings (+ 1))
            destination = Public.GenericParameters (layer parameters) (Public.Settings (* 3))
        restored <- H.evalEither (NN.loadWeights (NN.extractWeights source) destination)
        let Public.Settings configure = Public.genericSettings restored
        configure 4 H.=== 12
        T.toList (NN.denseWeights (Public.genericLayer restored)) H.=== [0.25, 0.5],
      testProperty "manual sum traversal preserves the destination constructor and configuration" $ H.withTests 1 $ H.property $ do
        let source = Public.Enabled (T.fromListData @'[1] [5]) (Public.Settings (+ 1))
            destination = Public.Enabled (T.fromListData @'[1] [0]) (Public.Settings (* 2))
        restored <- H.evalEither (NN.loadWeights (NN.extractWeights source) destination)
        case restored of
          Public.Enabled tensor (Public.Settings configure) -> do
            T.toList tensor H.=== [5]
            configure 3 H.=== 6
          Public.Disabled _ -> H.failure
        rejects (Weights.MissingFields [["enabled"]]) (NN.loadWeights (NN.extractWeights (Public.Disabled (Public.Settings id))) destination),
      testProperty "empty structures contain no weights" $ H.withTests 1 $ H.property $ do
        map fst (Weights.weightsToList (NN.extractWeights ())) H.=== []
        restored <- H.evalEither (NN.loadWeights (NN.extractWeights ()) (7 :: Int))
        restored H.=== 7
    ]
