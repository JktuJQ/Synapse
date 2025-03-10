{- | Provides interface for creating and using neural network models.
-}


{- @TypeFamilies@ are needed to instantiate @Container@ typeclass.
-}

{-# LANGUAGE TypeFamilies #-}

{- @FlexibleContexts@, @FlexibleInstances@, @MultiParamTypeClasses@ are needed to define @Model@ typeclass.
-}

{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module Synapse.NN.Models
    ( -- * @Model@ typeclass
      Model (Training, toTrainingMode, toEvalMode)

      -- * Common for models
    , InputSize (InputSize)

      -- * @SequentialModel@ datatype
      
    , SequentialModel (SequentialModel, unSequentialModel)
    , buildSequentialModel
    ) where


import Synapse.NN.Layers.Layer(AbstractLayer(..), AbstractLayerM(..), Layer, LayerConfiguration)

import Synapse.Tensors (DType, SingletonOps(singleton))

import Data.Maybe (fromMaybe)
import Data.Functor ((<&>))

import qualified Data.Vector as V
import qualified Data.Vector.Mutable as MV


-- | @Model@ typeclass represents all neural network model that are able to be trained (mutated in a monadic context).
class (MV.PrimMonad m, AbstractLayerM (Training model m) m, AbstractLayer model) => Model model m where
    -- | Training version of a model.
    data Training model m a

    -- | Turns model to training mode.
    toTrainingMode :: model a -> m (Training model m a)
    -- | Turnes model that was in a training mode to evaluation mode.
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

layerPrefix :: String -> Int -> String
layerPrefix prefix i = prefix ++ "l" ++ show i ++ "w"

instance AbstractLayer SequentialModel where
    inputSize = inputSize . V.head . unSequentialModel
    outputSize = outputSize . V.head . unSequentialModel

    nParameters = V.foldl' (\parameters layer -> parameters + nParameters layer) 0 . unSequentialModel
    getParameters prefix =
        snd . V.foldl' (\(i, acc) layer -> (i + 1, acc ++ getParameters (layerPrefix prefix i) layer)) (1 :: Int, []) . unSequentialModel
    updateParameters model = SequentialModel . V.fromList . go (V.toList $ unSequentialModel model)
      where
        go [] _ = []
        go (layer:layers) parameters = let (x, parameters') = splitAt (nParameters layer) parameters
                                       in updateParameters layer x : go layers parameters'

    applyRegularizer prefix =
        snd . V.foldl' (\(i, mat) layer -> (i + 1, mat + applyRegularizer (layerPrefix prefix i) layer)) (1 :: Int, singleton 0) . unSequentialModel

    symbolicForward prefix input =
        snd . V.foldl' (\(i, mat) layer -> (i + 1, symbolicForward (layerPrefix prefix i) mat layer)) (1 :: Int, input) . unSequentialModel


instance MV.PrimMonad m => Model SequentialModel m where
    -- | @TrainingSequentialModel@ newtype wraps mutable vector of layers, and so it represents mutable version of @SequentialModel@.
    newtype Training SequentialModel m a = TrainingSequentialModel
        { unTrainingSequentialModel :: MV.MVector (MV.PrimState m) (Layer a)  -- ^ Returns layers of @TrainingSequentialModel@.
        }
    
    toTrainingMode (SequentialModel layers) = V.thaw layers <&> TrainingSequentialModel
    toEvalMode (TrainingSequentialModel layers) = V.freeze layers <&> SequentialModel

type instance DType (Training SequentialModel m a) = a

instance MV.PrimMonad m => AbstractLayerM (Training SequentialModel m) m where
    inputSizeM (TrainingSequentialModel layers) = MV.read layers 0 <&> inputSize
    outputSizeM (TrainingSequentialModel layers) = MV.read layers 0 <&> outputSize

    nParametersM = MV.foldl' (\parameters layer -> parameters + nParameters layer) 0 . unTrainingSequentialModel
    getParametersM prefix =
        fmap snd . MV.foldl' (\(i, acc) layer -> (i + 1, acc ++ getParameters (layerPrefix prefix i) layer)) (1 :: Int, []) . unTrainingSequentialModel
    updateParametersM model p = go (unTrainingSequentialModel model) p >> return model
      where
        go layers parameters
            | MV.null layers = return ()
            | otherwise      = do
                layer <- MV.read layers 0
                let (x, parameters') = splitAt (nParameters layer) parameters
                _ <- MV.write layers 0 (updateParameters layer x)
                go (MV.tail layers) parameters'

    applyRegularizerM prefix =
        fmap snd . MV.foldl' (\(i, mat) layer -> (i + 1, mat + applyRegularizer (layerPrefix prefix i) layer)) (1 :: Int, singleton 0) . unTrainingSequentialModel

    symbolicForwardM prefix input =
        fmap snd . MV.foldl' (\(i, mat) layer -> (i + 1, symbolicForward (layerPrefix prefix i) mat layer)) (1 :: Int, input) . unTrainingSequentialModel