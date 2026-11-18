{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

{- | Materialized tensor weights keyed by structural field paths.
'Weights' stores tensor names, dimensions, element types, and values independently of a file format.

= Extracting and restoring weights

'extractWeights' visits materialized fields of a 'TensorStructure'.
Run staged parameters first to obtain materialized values:

@
import Synapse.NN.Layers.Dense (Dense (..))
import qualified Data.Map.Strict as Map
import qualified Synapse.NN.Weights as W
import qualified Synapse.Tensor as T

parameters :: Dense 2 1 T.Tensor Float
parameters = Dense (T.fromList \@'[2, 1] [0.25, 0.5]) (T.zeros \@'[1])

template = T.run T.CPU parameters
weights = W.extractWeights template
restored = W.loadWeights weights template
@

'loadWeights' replaces tensors in an existing destination structure.
The destination supplies field paths, shapes, and element types.
Paths must match exactly; loading neither converts dtypes nor changes tensor layouts.
Structures contain only tensors or nested tensor structures.

Use "Synapse.NN.Weights.Safetensors" to encode and decode 'Weights' or read and write files.
Other codecs can inspect entries with 'withWeightTensor'
and reconstruct them with 'weightTensorFromView'.

= Adapting external weights

A file's names and tensor layouts may differ from the destination model.
Rename paths and transform entries before calling 'loadWeights'.
Use 'weightsEntries' to work with the dictionary using "Data.Map.Strict".
'weightTensorToTensorData' checks an entry against a chosen shape and dtype.
Use ordinary tensor operations to reshape, slice, or convert it, then materialize and wrap the result.
-}
module Synapse.NN.Weights
    ( -- * Tensor fields
      FieldPath

      -- * Tensor entries
    , WeightTensor
    , weightTensorShape
    , weightTensorType
    , weightTensorFromTensorData
    , weightTensorToTensorData

      -- * Element views
    , WeightTensorView (..)
    , weightTensorView
    , weightTensorFromView
    , withWeightTensor

      -- * Named weights
    , Weights (..)

      -- * Extraction and restoration
    , WeightsError (..)
    , extractWeights
    , loadWeights
    ) where

import Data.Proxy (Proxy (..))
import Data.Type.Equality ((:~:) (Refl))
import Data.Typeable (TypeRep, Typeable, eqT, typeRep)

import Synapse.Tensor.Representation (DType, TensorData (..))
import Synapse.Tensor.Structure (ElementConstraints, FieldPath, TensorStructure, WithTensor)

import Synapse.Tensor.Shapes.Backend (backendShape)
import Synapse.Tensor.Shapes.Dimensions (Shape (..))
import qualified Synapse.Tensor.Structure as Structure

import qualified Data.Array.Accelerate as A
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set

{- | A host tensor with runtime dimensions and an existential element type.

Elements are stored in row-major order; a scalar has dimensions @[]@ and one element.
Construct entries from 'TensorData' with 'weightTensorFromTensorData' or from runtime shapes with the indexed/list builders.
-}
data WeightTensor = forall a. (DType a, Typeable a) => WeightTensor [Int] (A.Vector a)

-- | Runtime dimensions in axis order; scalars have no dimensions.
weightTensorShape
    :: WeightTensor
    -> [Int]
weightTensorShape (WeightTensor dimensions _) = dimensions

{- | Inspect the exact Haskell element type for dispatch in a codec.

Use codec-defined dtype identifiers in files; the printed 'TypeRep' is not a portable file identifier.
-}
weightTensorType
    :: WeightTensor
    -> TypeRep
weightTensorType (WeightTensor _ (_ :: A.Vector a)) = typeRep (Proxy @a)

-- | Wrap materialized tensor data, sharing its elements and retaining the type-level dimensions.
weightTensorFromTensorData
    :: forall sh a
     . (Shape sh, DType a, Typeable a)
    => TensorData sh a
    -> WeightTensor
weightTensorFromTensorData (TensorData values) = WeightTensor (shape @sh) (A.arrayReshape (A.Z A.:. shapeSize @sh) values)

-- | Check an entry against a destination tensor, retaining its field path in errors.
restoreTensor
    :: forall sh a
     . (Shape sh, DType a, Typeable a)
    => FieldPath
    -> WeightTensor
    -> Either WeightsError (TensorData sh a)
restoreTensor path entry@(WeightTensor dimensions (values :: A.Vector b))
    | shape @sh /= dimensions = Left (ShapeMismatch path (shape @sh) dimensions)
    | otherwise = case eqT @a @b of
        Just Refl -> Right (TensorData (A.arrayReshape (backendShape (Proxy @sh)) values))
        Nothing -> Left (ElementTypeMismatch path (typeRep (Proxy @a)) (weightTensorType entry))

-- | Recover typed tensor data after checking the exact dimensions and element type.
weightTensorToTensorData
    :: forall sh a
     . (Shape sh, DType a, Typeable a)
    => WeightTensor
    -> Either WeightsError (TensorData sh a)
weightTensorToTensorData = restoreTensor []

{- | Tensor dimensions and indexed row-major elements.

The index function must accept offsets in @[0, product shape)@.
Views returned by 'withWeightTensor' share storage and reject out-of-range offsets.
-}
data WeightTensorView a = WeightTensorView
    { weightTensorViewShape :: [Int]
    , weightTensorViewAt :: Int -> a
    }

-- | Create a view of 'WeightTensor'.
weightTensorView
    :: forall a
     . Typeable a
    => WeightTensor
    -> Maybe (WeightTensorView a)
weightTensorView tensor = withWeightTensor tensor $ \(values :: WeightTensorView b) -> case eqT @a @b of
    Just Refl -> Just values
    Nothing -> Nothing

{- | Materialize a view into a weight tensor without constructing an intermediate list.

Invalid dimensions return 'InvalidShape' before the index function is called.
The index function must provide every element; its exceptions are not caught.
-}
weightTensorFromView
    :: (DType a, Typeable a)
    => WeightTensorView a
    -> Either WeightsError WeightTensor
weightTensorFromView (WeightTensorView dimensions readElement) = do
    count <- checkedSize dimensions
    pure $ WeightTensor dimensions $ A.fromFunction (A.Z A.:. count) (\(A.Z A.:. index) -> readElement index)

{- | Inspect dimensions and elements while retaining dtype and runtime type evidence.

The callback must accept every supported element type; use 'Typeable' to dispatch to a codec's encoders.
The view shares the original array and does not allocate an intermediate element list.
-}
withWeightTensor
    :: WeightTensor
    -> (forall a. (DType a, Typeable a) => WeightTensorView a -> r)
    -> r
withWeightTensor (WeightTensor dimensions values) inspect =
    inspect $ WeightTensorView dimensions readElement
  where
    count = A.arraySize values
    readElement index
        | index < 0 || index >= count = error "Synapse.NN.Weights.weightTensorViewAt: index out of bounds"
        | otherwise = A.linearIndexArray values index

-- | Errors constructing, adapting, or restoring tensor weights.
data WeightsError
    = -- | Multiple tensor entries have the same field path.
      DuplicateField FieldPath
    | -- | Destination paths absent from the dictionary, in lexicographic order.
      MissingFields [FieldPath]
    | -- | Dictionary paths absent from the destination, in lexicographic order.
      UnexpectedFields [FieldPath]
    | -- | Field path, destination dimensions, and stored dimensions.
      ShapeMismatch FieldPath [Int] [Int]
    | -- | Field path, destination element type, and stored element type.
      ElementTypeMismatch FieldPath TypeRep TypeRep
    | -- | Negative dimensions or an element count exceeding the range of 'Int'.
      InvalidShape [Int]
    deriving (Eq, Show)

-- | Validate a dynamic shape and calculate its representable element count.
checkedSize
    :: [Int]
    -> Either WeightsError Int
checkedSize dimensions
    | any (< 0) dimensions || size > toInteger (maxBound :: Int) = Left (InvalidShape dimensions)
    | otherwise = Right (fromInteger size)
  where
    size = product (map toInteger dimensions)

-- | A dictionary from unique field paths to materialized tensor entries.
newtype Weights = Weights
    { weightsEntries :: Map.Map FieldPath WeightTensor
    }

{- | Collect materialized tensor fields into a dictionary, sharing their element storage.

'TensorStructure' instances must supply unique paths; this function does not check for collisions.
-}
extractWeights
    :: forall p
     . (TensorStructure p, ElementConstraints Typeable p)
    => WithTensor TensorData p
    -> Weights
extractWeights parameters =
    Weights $
        Structure.foldElements @Typeable @p @TensorData
            (\path entries tensor -> Map.insertWith (\_ old -> old) path (weightTensorFromTensorData tensor) entries)
            Map.empty
            parameters

{- | Replace tensor fields in a destination structure.

Missing paths are reported first, followed by unexpected paths; both lists use lexicographic order.
If paths match, tensors are checked in traversal order for exact shapes and Haskell element types.
No dtype conversion, reshaping, or partial loading is performed.
On success, tensor storage is shared with the dictionary and all names, shapes, and types have been checked.
-}
loadWeights
    :: forall p
     . (TensorStructure p, ElementConstraints Typeable p)
    => Weights
    -> WithTensor TensorData p
    -> Either WeightsError (WithTensor TensorData p)
loadWeights (Weights entries) parameters
    | not (null missing) = Left (MissingFields missing)
    | not (null unexpected) = Left (UnexpectedFields unexpected)
    | otherwise = Structure.traverseElements @Typeable @p @TensorData @TensorData restore parameters
  where
    expected =
        Structure.foldElements @Typeable @p @TensorData
            (\path paths _ -> Set.insert path paths)
            Set.empty
            parameters
    stored = Map.keysSet entries
    missing = Set.toAscList (Set.difference expected stored)
    unexpected = Set.toAscList (Set.difference stored expected)
    restore
        :: forall sh a
         . (Shape sh, DType a, Typeable a)
        => FieldPath
        -> TensorData sh a
        -> Either WeightsError (TensorData sh a)
    restore path _ = case Map.lookup path entries of
        Nothing -> Left (MissingFields [path])
        Just entry -> restoreTensor path entry
