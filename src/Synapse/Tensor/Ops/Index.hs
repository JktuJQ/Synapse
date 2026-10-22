{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Static element indexing and multidimensional region selection.
module Synapse.Tensor.Ops.Index (Slicing (..), Indexing (..), Coordinates (..), Indexable (..), coordinate, index, slice) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Axes (CanSelectAxes)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM, backendCoordinates, backendIndex, backendShape, shape)
import Synapse.Tensor.Shapes.Index
  ( CanIndex,
    CanSlice,
    Index,
    Slice,
    SliceAxis (..),
    SliceShape,
    coordinates,
    sliceAxes,
  )
import Synapse.Tensor.Type (Element, Tensor (..), TensorElement (..), TensorIndex (..))
import Prelude hiding ((!!))
import qualified Prelude as P

-- | Statically shaped tensor regions.
class (DType a) => Slicing t a where
  -- | Select a region using one descriptor per axis.
  -- At removes an axis; ranges preserve it.
  -- Ranges have exclusive stops and strict bounds.
  -- A forward range whose stop precedes its start is empty.
  -- RangeReverse reverses the coordinates selected by the corresponding Range, without changing the output shape.
  sliceOp ::
    forall spec sh.
    (Shape sh, Shape (SliceShape spec sh), Slice spec, CanSlice spec sh) =>
    Proxy spec ->
    t sh a ->
    t (SliceShape spec sh) a

instance (DType a) => Slicing Tensor a where
  sliceOp ::
    forall spec sh.
    (Shape sh, Shape (SliceShape spec sh), Slice spec, CanSlice spec sh) =>
    Proxy spec ->
    Tensor sh a ->
    Tensor (SliceShape spec sh) a
  sliceOp _ (Tensor acc) = Tensor $ A.backpermute targetShape sourceIndex acc
    where
      targetShape = A.constant (backendShape (Proxy @(SliceShape spec sh)))
      targetDims = shape @(SliceShape spec sh)

      sourceIndex :: A.Exp (ShapeToDIM (SliceShape spec sh)) -> A.Exp (ShapeToDIM sh)
      sourceIndex ix = backendIndex @sh (sliceCoordinates (sliceAxes @spec) targetDims (backendCoordinates @(SliceShape spec sh) ix))

-- | Select a region using one descriptor per axis.
-- At removes an axis; ranges preserve it.
-- Ranges have exclusive stops and strict bounds.
-- A forward range whose stop precedes its start is empty.
-- RangeReverse reverses the coordinates selected by the corresponding Range, without changing the output shape.
slice ::
  forall {t} spec sh a.
  (Slicing t a, Shape sh, Shape (SliceShape spec sh), Slice spec, CanSlice spec sh) =>
  t sh a ->
  t (SliceShape spec sh) a
slice = sliceOp @t @a @spec @sh Proxy

-- | Shape-checked access to individual tensor elements.
class (DType a) => Indexing t a where
  -- | Read one element as an expression, without executing the tensor.
  -- For example, @index \@'[1, 2] matrix@.
  -- Use scalar to materialize the expression.
  indexOp ::
    forall ix sh.
    (Shape sh, Index ix, CanIndex ix sh) =>
    Proxy ix ->
    t sh a ->
    Element t a

instance (DType a) => Indexing Tensor a where
  indexOp ::
    forall ix sh.
    (Shape sh, Index ix, CanIndex ix sh) =>
    Proxy ix ->
    Tensor sh a ->
    TensorElement a
  indexOp _ (Tensor acc) = TensorElement (acc A.! backendIndex @sh (map A.constant (coordinates @ix)))

-- | Read one element as an expression, without executing the tensor.
-- For example, @index \@'[1, 2] matrix@.
-- Use scalar to materialize the expression.
index ::
  forall {t} ix sh a.
  (Indexing t a, Shape sh, Index ix, CanIndex ix sh) =>
  t sh a ->
  Element t a
index = indexOp @t @a @ix @sh Proxy

-- | Coordinate expressions of complete tensor indices.
class Coordinates element where
  -- | Read a coordinate of a complete index.
  -- Axes are numbered from zero, from left to right.
  -- The selected axis must exist in the index shape.
  coordinateOp ::
    forall axis sh.
    (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) =>
    Proxy axis ->
    TensorIndex sh ->
    element Int

instance Coordinates TensorElement where
  coordinateOp ::
    forall axis sh.
    (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) =>
    Proxy axis ->
    TensorIndex sh ->
    TensorElement Int
  coordinateOp _ (TensorIndex ix) = TensorElement (backendCoordinates @sh ix P.!! axisNumber)
    where
      axisNumber = fromInteger (toInteger (natVal (Proxy @axis)))

-- | Read a coordinate of a complete index.
-- Axes are numbered from zero, from left to right.
-- The selected axis must exist in the index shape.
coordinate ::
  forall {element} axis sh.
  (Coordinates element, KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) =>
  TensorIndex sh ->
  element Int
coordinate = coordinateOp @element @axis @sh Proxy

-- | Read a coordinate or tensor element using a shape-checked index.
-- Use @ix !! Proxy \@0@ for a coordinate and @tensor !! Proxy \@'[1, 2]@ for a static element.
-- A 'TensorIndex' of the same shape can also be used directly as @tensor !! ix@.
class Indexable source selector result | source selector -> result where
  -- | Build an element expression without executing the tensor.
  (!!) :: source -> selector -> result

infixl 9 !!

instance (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) => Indexable (TensorIndex sh) (Proxy axis) (TensorElement Int) where
  ix !! _ = coordinate @axis ix

instance (Shape sh, Index ix, CanIndex ix sh, DType a) => Indexable (Tensor sh a) (Proxy ix) (TensorElement a) where
  tensor !! _ = index @ix tensor

instance (Shape sh, DType a) => Indexable (Tensor sh a) (TensorIndex sh) (TensorElement a) where
  Tensor acc !! TensorIndex ix = TensorElement (acc A.! ix)

-- | Map retained coordinates directly to source coordinates, inserting fixed coordinates for removed axes.
-- Retained dimensions determine the starting coordinate of reversed ranges.
sliceCoordinates :: [SliceAxis Int] -> [Int] -> [A.Exp Int] -> [A.Exp Int]
sliceCoordinates [] [] [] = []
sliceCoordinates (At coord : rest) dims coords =
  A.constant coord : sliceCoordinates rest dims coords
sliceCoordinates (selection : rest) (dim : dims) (coord : coords) =
  sourceCoordinate : sliceCoordinates rest dims coords
  where
    sourceCoordinate = case selection of
      All -> coord
      Range begin _ step -> A.constant begin + A.constant step * coord
      RangeReverse begin _ step -> A.constant (begin + max 0 (dim - 1) * step) - A.constant step * coord
sliceCoordinates _ _ _ = error "Synapse.Tensor.slice: inconsistent selection rank"
