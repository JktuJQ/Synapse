{- | Provides interface for creating and using neural network models.
-}


module Synapse.ML.Models
    ( -- * Common
      Input (Input)

      -- * @SequentialModel@ datatype
      
    , SequentialModel (SequentialModel, unSequentialModel)
    , buildSequentialModel
    ) where


import Synapse.ML.Layers (AbstractLayer(..), Layer, LayerConfiguration)

import Data.Maybe (fromMaybe)

import qualified Data.Vector as V


-- | @Input@ newtype wraps @Int@ - amount of features of input that the model should support (@Input 3@ means that model supports any matrix with size (x, 3)).
newtype Input = Input Int

-- | @SequentialModel@ datatype represents any model grouping layers linearly.
newtype SequentialModel a = SequentialModel
    { unSequentialModel :: V.Vector (Layer a)  -- ^ Returns layers of @SequentialModel@
    }

-- | Builds sequential model using input size and layer configurations to ensure that layers are compatible with each other.
buildSequentialModel :: Input -> [LayerConfiguration (Layer a)] -> SequentialModel a
buildSequentialModel (Input input) layerConfigs = SequentialModel $ V.fromList $ go input layerConfigs
  where
    go _ [] = []
    go prevSize (l:ls) = let layer = l prevSize
                             output = outputSize layer
                             output' = fromMaybe prevSize output
                         in layer : go output' ls


instance Functor SequentialModel where
    fmap f = SequentialModel . fmap (fmap f) . unSequentialModel

instance AbstractLayer SequentialModel where
    inputSize = inputSize . V.head . unSequentialModel
    outputSize = outputSize . V.head . unSequentialModel

    getParameters = V.foldl' (\acc x -> acc ++ getParameters x) [] . unSequentialModel
    updateParameters (SequentialModel layers) parameters = SequentialModel $ fmap (`updateParameters` parameters) layers

    symbolicForward prefix (SequentialModel layers) input = V.foldl' (flip $ symbolicForward prefix) input layers
    forward (SequentialModel layers) input = V.foldl' (flip forward) input layers
