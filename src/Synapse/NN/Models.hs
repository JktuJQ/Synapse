{- | Provides interface for creating and using neural network models.
-}


-- 'TypeFamilies' are needed to instantiate 'Synapse.Tensors.DType'.
{-# LANGUAGE TypeFamilies #-}


module Synapse.NN.Models
    ( -- * Common for models
      InputSize (InputSize)

      -- * 'SequentialModel' datatype
      
    , SequentialModel (SequentialModel, unSequentialModel)
    , buildSequentialModel

    , layerPrefix
    ) where


import Synapse.Tensors (DType, SingletonOps(singleton))

import Synapse.Autograd (SymbolIdentifier (SymbolIdentifier))

import Synapse.NN.Layers.Layer(AbstractLayer(..), Layer, LayerConfiguration)

import Data.Maybe (fromMaybe)
import Data.Foldable (foldl')


-- | 'InputSize' newtype wraps 'Int' - amount of features of input that the model should support (@InputSize 3@ means that model supports any matrix with size (x, 3)).
newtype InputSize = InputSize Int


-- | 'SequentialModel' datatype represents any model grouping layers linearly.
newtype SequentialModel a = SequentialModel
    { unSequentialModel :: [Layer a]  -- ^ Returns layers of 'SequentialModel'.
    }

instance Show a => Show (SequentialModel a) where
    show (SequentialModel layers) = go 1 layers
      where
        go _ [] = ""
        go i (x:xs) = "Layer " ++ show i ++ " parameters: " ++ show (getParameters (layerPrefix mempty i) x) ++ ";\n" ++ go (i + 1) xs

-- | Builds sequential model using input size and layer configurations to ensure that layers are compatible with each other.
buildSequentialModel :: InputSize -> [LayerConfiguration (Layer a)] -> SequentialModel a
buildSequentialModel (InputSize i) layerConfigs = SequentialModel $ go i layerConfigs
  where
    go _ [] = []
    go prevSize (l:ls) = let layer = l prevSize
                             outputMaybe = outputSize layer
                             output = fromMaybe prevSize outputMaybe
                         in layer : go output ls

type instance DType (SequentialModel a) = a


-- | Forms prefix for layers according to 'Synapse.NN.Layers.Layer.AbstractLayer' requirements.
layerPrefix :: SymbolIdentifier -> Int -> SymbolIdentifier
layerPrefix prefix i = mconcat [prefix, SymbolIdentifier "l", SymbolIdentifier (show i), SymbolIdentifier "w"]

instance AbstractLayer SequentialModel where
    inputSize = inputSize . head . unSequentialModel
    outputSize = outputSize . head . unSequentialModel

    nParameters = foldl' (\parameters layer -> parameters + nParameters layer) 0 . unSequentialModel
    getParameters prefix =
        snd . foldl' (\(i, acc) layer -> (i + 1, acc ++ getParameters (layerPrefix prefix i) layer)) (1, []) . unSequentialModel
    updateParameters (SequentialModel model) = SequentialModel . go model
      where
        go [] _ = []
        go (layer:layers) parameters = let (x, parameters') = splitAt (nParameters layer) parameters
                                       in updateParameters layer x : go layers parameters'

    symbolicForward prefix input =
        snd . foldl' (\(i, (mat, loss)) layer -> let (mat', newLoss) = symbolicForward (layerPrefix prefix i) mat layer
                                                 in (i + 1, (mat', loss + newLoss)))
              (1, (input, singleton 0)) . unSequentialModel
