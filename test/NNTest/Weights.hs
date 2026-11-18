{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module NNTest.Weights (tests) where

import Control.Exception (ErrorCall, evaluate, try)
import Control.Monad (void)
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import Data.Typeable (cast, typeRep)
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified NNTest.Weights.Public as Public
import qualified NNTest.Weights.Safetensors as Safetensors
import qualified Synapse.NN as NN
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Structure as Structure
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

viewToList :: Weights.WeightTensorView a -> [a]
viewToList view = map (Weights.weightTensorViewAt view) [0 .. product (Weights.weightTensorViewShape view) - 1]

newtype Parameters t a = Parameters
    { layer :: NN.Dense 2 1 t a
    }
    deriving (Generic)

instance Structure.TensorStructure (Parameters T.Tensor a) where
    type WithTensor t (Parameters T.Tensor a) = Parameters t a

parameters :: Parameters T.TensorData Double
parameters = Parameters (NN.Dense (T.fromListData @'[2, 1] [0.25, 0.5]) (T.fromListData @'[1] [0.1]))

checkWeights :: H.MonadTest m => Parameters T.TensorData Double -> m ()
checkWeights restored = do
    T.toList (NN.denseWeights (layer restored)) H.=== [0.25, 0.5]
    T.toList (NN.denseBias (layer restored)) H.=== [0.1]

rejects :: H.MonadTest m => Weights.WeightsError -> Either Weights.WeightsError a -> m ()
rejects expected result = case result of
    Left actual -> actual H.=== expected
    Right _ -> H.failure

tests :: TestTree
tests =
    testGroup
        "weight state"
        [ Safetensors.tests
        , testProperty "dictionary entries can be renamed and selected" $ H.withTests 1 $ H.property $ do
            let tensor = Weights.weightTensorFromTensorData (T.fromListData @'[] [1 :: Double])
            let dictionary = Weights.Weights $ Map.fromList [(["a"], tensor), (["b"], tensor)]
            let renamed = Weights.Weights $ Map.mapKeys ("layer" :) $ Weights.weightsEntries dictionary
            map fst ((Map.toAscList . Weights.weightsEntries) renamed) H.=== [["layer", "a"], ["layer", "b"]]
            let selected = Weights.Weights $ Map.filterWithKey (\path _ -> path == ["layer", "a"]) $ Weights.weightsEntries renamed
            map fst ((Map.toAscList . Weights.weightsEntries) selected) H.=== [["layer", "a"]]
            let inserted = Weights.Weights $ Map.insert ["new"] tensor $ Map.delete ["layer", "a"] $ Weights.weightsEntries selected
            map fst ((Map.toAscList . Weights.weightsEntries) inserted) H.=== [["new"]]
        , testProperty "typed entries use ordinary tensor transformations" $ H.withTests 1 $ H.property $ do
            let source = Weights.weightTensorFromTensorData $ T.fromListData @'[3, 2] [1, 2, 3, 4, 5, 6 :: Double]
            matrix <- H.evalEither $ Weights.weightTensorToTensorData @'[3, 2] @Double source
            let transposed = T.run T.CPU $ T.transpose $ T.fromData matrix
                reshaped = T.run T.CPU $ T.reshape @'[6] $ T.transpose $ T.fromData matrix
                middle = T.run T.CPU $ T.slice @'[ 'T.Range 1 2 1, 'T.All] $ T.fromData matrix
            T.toList transposed H.=== [1, 3, 5, 2, 4, 6]
            T.toList reshaped H.=== [1, 3, 5, 2, 4, 6]
            T.toList middle H.=== [3, 4]
            rejects (Weights.ShapeMismatch [] [2, 3] [3, 2]) $ Weights.weightTensorToTensorData @'[2, 3] @Double source
            rejects (Weights.ElementTypeMismatch [] (typeRep (Proxy @Float)) (typeRep (Proxy @Double))) $
                Weights.weightTensorToTensorData @'[3, 2] @Float source
        , testProperty "typed entries support scalars empty tensors and round trips" $ H.withTests 1 $ H.property $ do
            let scalar = Weights.weightTensorFromTensorData $ T.fromListData @'[] [True]
            same <- H.evalEither $ Weights.weightTensorToTensorData @'[] @Bool scalar
            T.toList same H.=== [True]
            fmap viewToList (Weights.weightTensorView @Double scalar) H.=== Nothing
            let empty = Weights.weightTensorFromTensorData $ T.fromListData @'[2, 0] ([] :: [Double])
            emptyData <- H.evalEither $ Weights.weightTensorToTensorData @'[2, 0] @Double empty
            T.toList emptyData H.=== []
            Weights.weightTensorShape (Weights.weightTensorFromTensorData emptyData) H.=== [2, 0]
            restored <-
                H.evalEither $ Weights.weightTensorToTensorData @'[] @Bool $ Weights.weightTensorFromTensorData same
            T.toList restored H.=== [True]
        , testProperty "nested tensor records round trip with selector paths" $ H.withTests 1 $ H.property $ do
            let dictionary = NN.extractWeights parameters
                destination =
                    Parameters
                        (NN.Dense (T.fromListData @'[2, 1] [0, 0]) (T.fromListData @'[1] [0]))
            map fst ((Map.toAscList . Weights.weightsEntries) dictionary)
                H.=== [["layer", "denseBias"], ["layer", "denseWeights"]]
            restored <- H.evalEither (NN.loadWeights dictionary destination)
            checkWeights restored
        , testProperty "external codecs can reconstruct entries without static shapes" $ H.withTests 1 $ H.property $ do
            let encodeDecode (path, tensor) = Weights.withWeightTensor tensor $ \values ->
                    (,) path
                        <$> Weights.weightTensorFromView values
            entries <-
                H.evalEither (traverse encodeDecode ((Map.toAscList . Weights.weightsEntries) (NN.extractWeights parameters)))
            let dictionary = Weights.Weights $ Map.fromList $ reverse entries
            restored <- H.evalEither (NN.loadWeights dictionary parameters)
            checkWeights restored
        , testProperty "entry inspection exposes shape, type, and typed row-major elements" $ H.withTests 1 $ H.property $ do
            let tensor = Weights.weightTensorFromTensorData (T.fromListData @'[2, 1] [1, 2 :: Double])
            Weights.weightTensorShape tensor H.=== [2, 1]
            Weights.weightTensorType tensor H.=== typeRep (Proxy @Double)
            Weights.withWeightTensor tensor (\values -> cast (viewToList values) :: Maybe [Double])
                H.=== Just [1, 2]
        , testProperty "element views support indexed reads and explicit list conversion" $ H.withTests 1 $ H.property $ do
            let tensor = Weights.weightTensorFromTensorData (T.fromListData @'[2, 2] [10, 20, 30, 40 :: Int])
                inspect values =
                    ( product (Weights.weightTensorViewShape values)
                    , cast [Weights.weightTensorViewAt values 3, Weights.weightTensorViewAt values 0] :: Maybe [Int]
                    , cast (viewToList values) :: Maybe [Int]
                    )
            Weights.withWeightTensor tensor inspect H.=== (4, Just [40, 10], Just [10, 20, 30, 40])
        , testProperty "indexed views reject negative, past-end and empty indices" $ H.withTests 1 $ H.property $ do
            let rejectsIndex source index = do
                    result <- H.evalIO $ Weights.withWeightTensor source $ \values -> try @ErrorCall (void (evaluate (Weights.weightTensorViewAt values index)))
                    case result of
                        Left _ -> pure ()
                        Right _ -> H.failure
                tensor = Weights.weightTensorFromTensorData (T.fromListData @'[1] [1 :: Double])
            rejectsIndex tensor (-1)
            rejectsIndex tensor 1
            rejectsIndex (Weights.weightTensorFromTensorData (T.fromListData @'[0] ([] :: [Double]))) 0
        , testProperty "indexed construction handles row-major arrays, scalars and empty shapes" $ H.withTests 1 $ H.property $ do
            tensor <-
                H.evalEither
                    (Weights.weightTensorFromView (Weights.WeightTensorView [2, 3] (\index -> fromIntegral (index * 2) :: Double)))
            let dictionary = Weights.Weights $ Map.singleton [] tensor
            restored <- H.evalEither (NN.loadWeights dictionary (T.fromListData @'[2, 3] (replicate 6 (0 :: Double))))
            T.toList restored H.=== [0, 2, 4, 6, 8, 10]
            scalar <- H.evalEither (Weights.weightTensorFromView (Weights.WeightTensorView [] (== 0)))
            Weights.withWeightTensor
                scalar
                ( \values -> (product (Weights.weightTensorViewShape values), cast (Weights.weightTensorViewAt values 0) :: Maybe Bool)
                )
                H.=== (1, Just True)
            empty <-
                H.evalEither
                    (Weights.weightTensorFromView (Weights.WeightTensorView [2, 0] (\_ -> error "empty source was read" :: Double)))
            Weights.withWeightTensor
                empty
                (\values -> (product (Weights.weightTensorViewShape values), length (viewToList values)))
                H.=== (0, 0)
            rejects
                (Weights.InvalidShape [-1])
                (Weights.weightTensorFromView (Weights.WeightTensorView [-1] (\_ -> error "invalid source was read" :: Double)))
        , testProperty "a view can be transformed and materialized" $ H.withTests 1 $ H.property $ do
            let source = Weights.weightTensorFromTensorData (T.fromListData @'[2, 2] [1, 2, 3, 4 :: Double])
            view <- H.evalMaybe $ Weights.weightTensorView @Double source
            let reversed =
                    view
                        { Weights.weightTensorViewShape = [4]
                        , Weights.weightTensorViewAt = \i -> Weights.weightTensorViewAt view (3 - i)
                        }
            result <- H.evalEither $ Weights.weightTensorFromView reversed
            restored <- H.evalEither $ Weights.weightTensorToTensorData @'[4] @Double result
            T.toList restored H.=== [4, 3, 2, 1]
        , testProperty "scalars, empty tensors, mixed element types and tuple paths round trip" $ H.withTests 1 $ H.property $ do
            let source = (T.fromListData @'[] [True], T.fromListData @'[0, 2] ([] :: [Float]), T.fromListData @'[2] [1, 2 :: Int])
                destination = (T.fromListData @'[] [False], T.fromListData @'[0, 2] ([] :: [Float]), T.fromListData @'[2] [0, 0 :: Int])
                dictionary = NN.extractWeights source
            map fst ((Map.toAscList . Weights.weightsEntries) dictionary) H.=== [["0"], ["1"], ["2"]]
            (scalar, empty, integers) <- H.evalEither (NN.loadWeights dictionary destination)
            T.toList scalar H.=== [True]
            T.toList empty H.=== []
            T.toList integers H.=== [1, 2]
        , testProperty "loading rejects missing and unexpected fields" $ H.withTests 1 $ H.property $ do
            let entries = (Map.toAscList . Weights.weightsEntries) (NN.extractWeights parameters)
            let incomplete = Weights.Weights $ Map.fromList $ drop 1 entries
            rejects (Weights.MissingFields [["layer", "denseBias"]]) (NN.loadWeights incomplete parameters)
            let extra =
                    Weights.Weights $
                        Map.fromList ((["extra"], Weights.weightTensorFromTensorData (T.fromListData @'[] [0 :: Double])) : entries)
            rejects (Weights.UnexpectedFields [["extra"]]) (NN.loadWeights extra parameters)
        , testProperty "loading rejects shapes rather than reshaping equal-sized arrays" $ H.withTests 1 $ H.property $ do
            let source = NN.extractWeights (T.fromListData @'[2] [1, 2 :: Double])
                destination = T.fromListData @'[1, 2] [0, 0 :: Double]
            rejects (Weights.ShapeMismatch [] [1, 2] [2]) (NN.loadWeights source destination)
        , testProperty "loading rejects implicit element type conversion" $ H.withTests 1 $ H.property $ do
            let source = NN.extractWeights (T.fromListData @'[1] [1 :: Float])
                destination = T.fromListData @'[1] [0 :: Double]
            rejects
                (Weights.ElementTypeMismatch [] (typeRep (Proxy @Double)) (typeRep (Proxy @Float)))
                (NN.loadWeights source destination)
        , testProperty "dynamic entries validate dimensions" $ H.withTests 1 $ H.property $ do
            rejects
                (Weights.InvalidShape [-1])
                (Weights.weightTensorFromView $ Weights.WeightTensorView [-1] (\_ -> 0 :: Double))
            rejects
                (Weights.InvalidShape [maxBound, 2])
                (Weights.weightTensorFromView $ Weights.WeightTensorView [maxBound, 2] (\_ -> 0 :: Double))
            let scalar = Weights.weightTensorFromTensorData $ T.fromListData @'[] [3 :: Double]
            restored <-
                H.evalEither
                    (NN.loadWeights (NN.extractWeights (T.fromListData @'[] [3 :: Double])) (T.fromListData @'[] [0 :: Double]))
            Weights.weightTensorShape scalar H.=== []
            T.toList restored H.=== [3]
        , testProperty "public codecs and custom field names round trip without internal imports" $ H.withTests 1 $ H.property $ do
            let source = Public.Manual (layer parameters) (T.fromListData @'[] [True])
                destination =
                    Public.Manual
                        (NN.Dense (T.fromListData @'[2, 1] [0, 0]) (T.fromListData @'[1] [0]))
                        (T.fromListData @'[] [False])
            encoded <- H.evalEither (Public.encode (NN.extractWeights source))
            decoded <- H.evalEither (Public.decode encoded)
            map fst ((Map.toAscList . Weights.weightsEntries) decoded)
                H.=== [["enabled"], ["projection", "denseBias"], ["projection", "denseWeights"]]
            Public.Manual restored enabledTensor <- H.evalEither (NN.loadWeights decoded destination)
            T.toList (NN.denseWeights restored) H.=== [0.25, 0.5]
            T.toList (NN.denseBias restored) H.=== [0.1]
            T.toList enabledTensor H.=== [True]
        , testProperty "nested shape and dtype errors retain the complete path" $ H.withTests 1 $ H.property $ do
            let entries = (Map.toAscList . Weights.weightsEntries) (NN.extractWeights parameters)
                replace tensor =
                    Weights.Weights $
                        Map.fromList $
                            (["layer", "denseWeights"], tensor) : filter ((/= ["layer", "denseWeights"]) . fst) entries
            let wrongShape = replace $ Weights.weightTensorFromTensorData $ T.fromListData @'[2] [1, 2 :: Double]
            rejects (Weights.ShapeMismatch ["layer", "denseWeights"] [2, 1] [2]) $
                NN.loadWeights wrongShape parameters
            let wrongType = replace $ Weights.weightTensorFromTensorData $ T.fromListData @'[2, 1] [1, 2 :: Float]
            rejects
                (Weights.ElementTypeMismatch ["layer", "denseWeights"] (typeRep (Proxy @Double)) (typeRep (Proxy @Float)))
                $ NN.loadWeights wrongType parameters
        ]
