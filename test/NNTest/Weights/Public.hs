{-# LANGUAGE TypeFamilies #-}

module NNTest.Weights.Public
    ( Manual (..)
    , CodecError (..)
    , encode
    , decode
    ) where

import qualified Data.Map.Strict as Map
import Data.Typeable (Typeable, cast)
import GHC.Generics (Generic)
import qualified Synapse.NN as NN
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Structure as Structure
import Text.Read (readEither)

viewToList :: Weights.WeightTensorView a -> [a]
viewToList view = map (Weights.weightTensorViewAt view) [0 .. product (Weights.weightTensorViewShape view) - 1]

fromList :: (T.DType a, Typeable a) => [Int] -> [a] -> Either CodecError Weights.WeightTensor
fromList dimensions values
    | any (< 0) dimensions || product (map toInteger dimensions) /= toInteger (length values) =
        Left $ InvalidText "element count does not match shape"
    | otherwise =
        either (Left . InvalidWeights) Right $
            Weights.weightTensorFromView $
                Weights.WeightTensorView dimensions (values !!)

data Manual t = Manual (NN.Dense 2 1 t Double) (t '[] Bool)
    deriving (Generic)

instance Structure.TensorStructure (Manual T.Tensor) where
    type WithTensor t (Manual T.Tensor) = Manual t
    flatten (Manual (NN.Dense weights bias) enabled) =
        (
            ( Structure.TensorField ["projection", "denseWeights"] weights
            , Structure.TensorField ["projection", "denseBias"] bias
            )
        , Structure.TensorField ["enabled"] enabled
        )

data Payload = Doubles [Int] [Double] | Booleans [Int] [Bool] deriving (Read, Show)

data CodecError = UnsupportedElement | InvalidText String | InvalidWeights Weights.WeightsError
    deriving (Show)

encode :: Weights.Weights -> Either CodecError String
encode dictionary = show <$> traverse encodeEntry ((Map.toAscList . Weights.weightsEntries) dictionary)
  where
    encodeEntry (path, tensor) = Weights.withWeightTensor tensor $ \values ->
        let dimensions = Weights.weightTensorViewShape values
            payload = case cast values :: Maybe (Weights.WeightTensorView Double) of
                Just doubles -> Right (Doubles dimensions (viewToList doubles))
                Nothing -> case cast values :: Maybe (Weights.WeightTensorView Bool) of
                    Just booleans -> Right (Booleans dimensions (viewToList booleans))
                    Nothing -> Left UnsupportedElement
         in (,) path <$> payload

decode :: String -> Either CodecError Weights.Weights
decode input = do
    entries <- either (Left . InvalidText) Right (readEither input)
    let decodeEntry (path, payload) =
            let tensor = case payload of
                    Doubles dimensions values -> fromList dimensions values
                    Booleans dimensions values -> fromList dimensions values
             in (,) path <$> tensor
    decoded <- traverse decodeEntry entries
    pure $ Weights.Weights $ Map.fromList decoded
