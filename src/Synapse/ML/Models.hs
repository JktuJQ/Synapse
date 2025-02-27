{- | Provides interface for creating and using neural network models.
-}


module Synapse.ML.Models
    ( -- * Common
      InputSize (InputSize)

      -- * @SequentialModel@ datatype
      
    , SequentialModel (SequentialModel, unSequentialModel)
    , buildSequentialModel
    ) where


import Synapse.ML.Layers.Layer(AbstractLayer(..), Layer, LayerConfiguration)

import Synapse.LinearAlgebra (SingletonOps(singleton))

import Data.Foldable (foldl')
import Data.Maybe (fromMaybe)

import qualified Data.Vector as V


-- | @InputSize@ newtype wraps @Int@ - amount of features of input that the model should support (@InputSize 3@ means that model supports any matrix with size (x, 3)).
newtype InputSize = InputSize Int

-- | @SequentialModel@ datatype represents any model grouping layers linearly.
newtype SequentialModel a = SequentialModel
    { unSequentialModel :: V.Vector (Layer a)  -- ^ Returns layers of @SequentialModel@.
    }

-- | Builds sequential model using input size and layer configurations to ensure that layers are compatible with each other.
buildSequentialModel :: InputSize -> [LayerConfiguration (Layer a)] -> SequentialModel a
buildSequentialModel (InputSize i) layerConfigs = SequentialModel $ V.fromList $ go i layerConfigs
  where
    go _ [] = []
    go prevSize (l:ls) = let layer = l prevSize
                             outputMaybe = outputSize layer
                             output = fromMaybe prevSize outputMaybe
                         in layer : go output ls

instance AbstractLayer SequentialModel where
    inputSize = inputSize . V.head . unSequentialModel
    outputSize = outputSize . V.head . unSequentialModel

    getParameters = V.foldl' (\acc x -> acc ++ getParameters x) [] . unSequentialModel
    updateParameters (SequentialModel layers) parameters = SequentialModel $ fmap (`updateParameters` parameters) layers

    applyRegularizer prefix (SequentialModel layers) = foldl' (\loss layer -> loss + applyRegularizer prefix layer) (singleton 0) (V.toList layers)

    symbolicForward prefix (SequentialModel layers) input = V.foldl' (flip $ symbolicForward prefix) input layers
