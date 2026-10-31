{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Named, materialized tensor weights independent of a file format.
-- Extract weights with 'extractWeights' and restore them into an existing structure with 'loadWeights'.
module Synapse.NN.Weights
  ( WeightTensor,
    weightTensor,
    weightTensorShape,
    weightTensorType,
    TensorValues,
    valuesSize,
    valueAt,
    valuesToList,
    withWeightTensor,
    weightTensorFromFunction,
    weightTensorFromList,
    Weights,
    weightsToList,
    weightsFromList,
    WeightsError (..),
    extractWeights,
    loadWeights,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Functor.Const (Const (..))
import qualified Data.Map.Strict as Map
import Data.Proxy (Proxy (..))
import Data.Type.Equality ((:~:) (Refl))
import Data.Typeable (TypeRep, Typeable, eqT, typeRep)
import Synapse.Tensor.Fields (FieldPath, TensorFields (traverseTensors))
import Synapse.Tensor.Shapes.Dimensions (Shape (..), backendShape)
import Synapse.Tensor.Type (DType, TensorData (..))

-- | A tensor whose shape and element type are retained at runtime.
-- Storage is kept as a flat host array and shared with materialized tensors when possible.
data WeightTensor = forall a. (DType a, Typeable a) => WeightTensor [Int] (A.Vector a)

-- | Runtime dimensions in the same order as the original tensor.
weightTensorShape :: WeightTensor -> [Int]
weightTensorShape (WeightTensor dimensions _) = dimensions

-- | Runtime element type for dispatch in an external codec.
-- Codecs should define their own stable file identifiers rather than persist the printed representation.
weightTensorType :: WeightTensor -> TypeRep
weightTensorType (WeightTensor _ (_ :: A.Vector a)) = typeRep (Proxy @a)

-- | A read-only view of row-major elements backed by the original array.
-- Creating a view neither copies elements nor constructs an intermediate list.
-- Retaining a view keeps its source storage alive.
data TensorValues a = TensorValues !Int (Int -> a)

-- | Number of elements available in the view.
valuesSize :: TensorValues a -> Int
valuesSize (TensorValues count _) = count

-- | Read an element by its zero-based row-major index in constant time.
-- Indices outside @[0, valuesSize values)@ raise an error.
valueAt :: TensorValues a -> Int -> a
valueAt (TensorValues count readElement) index
  | index < 0 || index >= count = error "Synapse.NN.Weights.valueAt: index out of bounds"
  | otherwise = readElement index

-- | Explicitly produce a lazy row-major list for codecs that require lists.
-- Indexed codecs can use 'valueAt' directly to avoid the list allocation.
valuesToList :: TensorValues a -> [a]
valuesToList (TensorValues count readElement) = collect 0
  where
    collect index
      | index >= count = []
      | otherwise = readElement index : collect (index + 1)

-- | Inspect dimensions and a read-only element view with its element type available to a codec.
-- The view reads the existing array without a copy or an intermediate element list.
withWeightTensor :: WeightTensor -> (forall a. (DType a, Typeable a) => [Int] -> TensorValues a -> r) -> r
withWeightTensor (WeightTensor dimensions values) inspect = inspect dimensions (TensorValues (A.arraySize values) (A.linearIndexArray values))

-- | Errors encountered while constructing or restoring weights.
-- Shape and type mismatches report the destination expectation before the stored value.
data WeightsError
  = DuplicateField FieldPath
  | MissingFields [FieldPath]
  | UnexpectedFields [FieldPath]
  | ShapeMismatch FieldPath [Int] [Int]
  | ElementTypeMismatch FieldPath TypeRep TypeRep
  | InvalidShape [Int]
  | ElementCountMismatch Int Int
  deriving (Eq, Show)

-- | Validate a dynamic shape and calculate its representable element count.
checkedSize :: [Int] -> Either WeightsError Int
checkedSize dimensions
  | any (< 0) dimensions || size > toInteger (maxBound :: Int) = Left (InvalidShape dimensions)
  | otherwise = Right (fromInteger size)
  where
    size = product (map toInteger dimensions)

-- | Fill an entry directly from an indexed source without an intermediate list.
-- The reader is requested only at indices from zero to the element count minus one, in row-major order.
-- Empty tensors do not request any elements; scalars request index zero.
-- Invalid dimensions are rejected before reading the source.
-- The caller must validate source availability and decoding errors before supplying the reader.
weightTensorFromFunction :: forall a. (DType a, Typeable a) => [Int] -> (Int -> a) -> Either WeightsError WeightTensor
weightTensorFromFunction dimensions readElement = do
  count <- checkedSize dimensions
  pure (WeightTensor dimensions (A.fromFunction (A.Z A.:. count) (\(A.Z A.:. index) -> readElement index)))

-- | Construct an entry from a dynamic shape and finite row-major values.
-- Negative dimensions, sizes exceeding 'Int', and incorrect element counts are rejected.
-- Excess elements are reported at the first element beyond the expected count without traversing the remainder.
-- Scalars use an empty dimension list and exactly one value.
weightTensorFromList :: forall a. (DType a, Typeable a) => [Int] -> [a] -> Either WeightsError WeightTensor
weightTensorFromList dimensions values = do
  count <- checkedSize dimensions
  let checkCount 0 _ [] = Right (WeightTensor dimensions (A.fromList (A.Z A.:. count) values))
      checkCount 0 seen (_ : _) = Left (ElementCountMismatch count (seen + 1))
      checkCount _ seen [] = Left (ElementCountMismatch count seen)
      checkCount remaining seen (_ : rest) = checkCount (remaining - 1) (seen + 1) rest
  checkCount count 0 values

-- | Keep materialized weights without executing a backend or copying tensor elements.
-- The input must satisfy the shape invariant of 'TensorData'.
weightTensor :: forall sh a. (Shape sh, DType a, Typeable a) => TensorData sh a -> WeightTensor
weightTensor (TensorData values) = WeightTensor (shape @sh) (A.arrayReshape (A.Z A.:. shapeSize @sh) values)

-- | A unique mapping from field paths to tensor weights.
-- It contains no model functions, gradient history, or non-tensor configuration.
newtype Weights = Weights (Map.Map FieldPath WeightTensor)

-- | Entries in lexicographic path order, independent of their insertion order.
weightsToList :: Weights -> [(FieldPath, WeightTensor)]
weightsToList (Weights entries) = Map.toAscList entries

-- | Assemble decoded entries, rejecting duplicate paths instead of overwriting weights.
weightsFromList :: [(FieldPath, WeightTensor)] -> Either WeightsError Weights
weightsFromList = collect Map.empty
  where
    collect entries [] = Right (Weights entries)
    collect entries ((path, value) : rest)
      | Map.member path entries = Left (DuplicateField path)
      | otherwise = collect (Map.insert path value entries) rest

-- | Extract named weights without running a backend or copying tensor elements.
-- Ordinary fields are ignored; their values are not part of the dictionary.
-- User-defined 'TensorFields' instances must give every tensor a unique path.
extractWeights :: (TensorFields p) => p -> Weights
extractWeights parameters = Weights (getConst (traverseTensors (\path tensor -> Const (Map.singleton path (weightTensor tensor))) parameters))

-- | Replace weights in an existing structure, preserving all ordinary field values.
-- Missing or unexpected paths, mismatched shapes, and mismatched element types are errors.
-- A destination value supplies configuration and structural choices such as list lengths and optional fields.
-- The result is returned only when every weight has been checked.
loadWeights :: (TensorFields p) => Weights -> p -> Either WeightsError p
loadWeights (Weights entries) parameters
  | not (null missing) = Left (MissingFields missing)
  | not (null unexpected) = Left (UnexpectedFields unexpected)
  | otherwise = traverseTensors restore parameters
  where
    expected = getConst (traverseTensors (\path _ -> Const (Map.singleton path ())) parameters)
    missing = Map.keys (Map.difference expected entries)
    unexpected = Map.keys (Map.difference entries expected)
    restore :: forall sh a. (Shape sh, DType a, Typeable a) => FieldPath -> TensorData sh a -> Either WeightsError (TensorData sh a)
    restore path _ = case Map.lookup path entries of
      Nothing -> Left (MissingFields [path])
      Just entry@(WeightTensor dimensions (values :: A.Vector b))
        | shape @sh /= dimensions -> Left (ShapeMismatch path (shape @sh) dimensions)
        | otherwise -> case eqT @a @b of
            Just Refl -> Right (TensorData (A.arrayReshape (backendShape (Proxy @sh)) values))
            Nothing -> Left (ElementTypeMismatch path (typeRep (Proxy @a)) (weightTensorType entry))
