{- | Provides interface for creating and using neural network models.
-}


{- @TypeFamilies@ are needed to instantiate @Container@ typeclass.
-}

{-# LANGUAGE TypeFamilies #-}

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}


module Synapse.NN.Models
    ( -- * Common
      InputSize (InputSize)

      -- * @SequentialModel@ datatype
      
    , SequentialModel (SequentialModel, unSequentialModel)
    , buildSequentialModel
    ) where


import Synapse.NN.Layers.Layer(AbstractLayer(..), Layer, LayerConfiguration)

import Synapse.Tensors (DType, SingletonOps(singleton))

import Data.Maybe (fromMaybe)
import Data.Functor ((<&>))

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV


class MV.PrimMonad m => Model model m where
    data Training model m a

    toTrainingMode :: model a -> m (Training model m a)
    toEvalMode :: Training model m a -> m (model a)


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


type instance DType (SequentialModel a) = a

instance AbstractLayer SequentialModel where
    inputSize = inputSize . V.head . unSequentialModel
    outputSize = outputSize . V.head . unSequentialModel

    getParameters = V.foldl' (\acc x -> acc ++ getParameters x) [] . unSequentialModel
    updateParameters (SequentialModel layers) parameters = SequentialModel $ fmap (`updateParameters` parameters) layers

    applyRegularizer prefix (SequentialModel layers) = V.foldl' (\loss layer -> loss + applyRegularizer prefix layer) (singleton 0) layers

    symbolicForward prefix (SequentialModel layers) input = V.foldl' (flip $ symbolicForward prefix) input layers


instance MV.PrimMonad m => Model SequentialModel m where
    newtype Training SequentialModel m a = TrainingSequentialModel
        { unTrainingSequentialModel :: MV.MVector (MV.PrimState m) (Layer a)
        }
    
    toTrainingMode (SequentialModel layers) = V.thaw layers <&> TrainingSequentialModel
    toEvalMode (TrainingSequentialModel layers) = V.freeze layers <&> SequentialModel

type instance DType (Training SequentialModel m a) = a