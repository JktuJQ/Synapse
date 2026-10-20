{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Static element indexing and multidimensional region selection.
module Synapse.Tensor.Ops.Index (Indexable (..), coordinate, index, slice) where

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
import Synapse.Tensor.Type (Tensor (..), TensorElement (..), TensorIndex (..))
import Prelude hiding ((!!))
import qualified Prelude as P

-- | Read a coordinate of a complete index.
-- Axes are numbered from zero, from left to right.
-- The selected axis must exist in the index shape.
coordinate ::
  forall axis sh.
  (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) =>
  TensorIndex sh ->
  TensorElement Int
coordinate (TensorIndex ix) = TensorElement (backendCoordinates @sh ix P.!! axisNumber)
  where
    axisNumber = fromInteger (toInteger (natVal (Proxy @axis)))

-- | Read one element as an expression, without executing the tensor.
-- For example, @index \@'[1, 2] matrix@.
-- Use scalar to materialize the expression.
index ::
  forall ix sh a.
  (Shape sh, Index ix, CanIndex ix sh, DType a) =>
  Tensor sh a ->
  TensorElement a
index (Tensor acc) = TensorElement (acc A.! backendIndex @sh (map A.constant (coordinates @ix)))

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

-- | Select a region using one descriptor per axis.
-- At removes an axis; ranges preserve it.
-- Ranges have exclusive stops and strict bounds.
-- A forward range whose stop precedes its start is empty.
-- RangeReverse reverses the coordinates selected by the corresponding Range, without changing the output shape.
slice ::
  forall spec sh a.
  (Shape sh, Shape (SliceShape spec sh), Slice spec, CanSlice spec sh, DType a) =>
  Tensor sh a ->
  Tensor (SliceShape spec sh) a
slice (Tensor acc) = Tensor $ A.backpermute targetShape sourceIndex acc
  where
    targetShape = A.constant (backendShape (Proxy @(SliceShape spec sh)))
    targetDims = shape @(SliceShape spec sh)

    sourceIndex :: A.Exp (ShapeToDIM (SliceShape spec sh)) -> A.Exp (ShapeToDIM sh)
    sourceIndex ix = backendIndex @sh (sliceCoordinates (sliceAxes @spec) targetDims (backendCoordinates @(SliceShape spec sh) ix))
