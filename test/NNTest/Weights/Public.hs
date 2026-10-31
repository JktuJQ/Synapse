{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}

module NNTest.Weights.Public
  ( Settings (..),
    Manual (..),
    GenericParameters (..),
    Choice (..),
    CodecError (..),
    encode,
    decode,
  )
where

import Data.Typeable (cast)
import GHC.Generics (Generic)
import qualified Synapse.NN as NN
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.Tensor as T
import Text.Read (readEither)

newtype Settings = Settings (Int -> Int)

instance T.TensorFields Settings where
  traverseTensors _ = pure

data Manual = Manual (NN.Dense 2 1 T.TensorData Double) (T.TensorData '[] Bool) Settings

instance T.TensorFields Manual where
  traverseTensors visit (Manual layer enabled settings) =
    Manual
      <$> T.traverseTensors (\path -> visit ("projection" : path)) layer
      <*> visit ["enabled"] enabled
      <*> pure settings

data GenericParameters = GenericParameters
  { genericLayer :: NN.Dense 2 1 T.TensorData Double,
    genericSettings :: Settings
  }
  deriving (Generic)

data Choice = Disabled Settings | Enabled (T.TensorData '[1] Double) Settings

instance T.TensorFields Choice where
  traverseTensors _ (Disabled settings) = pure (Disabled settings)
  traverseTensors visit (Enabled tensor settings) = (`Enabled` settings) <$> visit ["enabled"] tensor

data Payload = Doubles [Int] [Double] | Booleans [Int] [Bool] deriving (Read, Show)

data CodecError = UnsupportedElement | InvalidText String | InvalidWeights Weights.WeightsError deriving (Show)

encode :: Weights.Weights -> Either CodecError String
encode dictionary = show <$> traverse encodeEntry (Weights.weightsToList dictionary)
  where
    encodeEntry (path, tensor) = Weights.withWeightTensor tensor $ \dimensions values ->
      let payload = case cast values :: Maybe (Weights.TensorValues Double) of
            Just doubles -> Right (Doubles dimensions (Weights.valuesToList doubles))
            Nothing -> case cast values :: Maybe (Weights.TensorValues Bool) of
              Just booleans -> Right (Booleans dimensions (Weights.valuesToList booleans))
              Nothing -> Left UnsupportedElement
       in (,) path <$> payload

decode :: String -> Either CodecError Weights.Weights
decode input = do
  entries <- either (Left . InvalidText) Right (readEither input)
  let decodeEntry (path, payload) =
        let tensor = case payload of
              Doubles dimensions values -> Weights.weightTensorFromList dimensions values
              Booleans dimensions values -> Weights.weightTensorFromList dimensions values
         in (,) path <$> either (Left . InvalidWeights) Right tensor
  decoded <- traverse decodeEntry entries
  either (Left . InvalidWeights) Right (Weights.weightsFromList decoded)
