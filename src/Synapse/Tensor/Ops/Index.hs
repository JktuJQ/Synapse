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
import Data.List (foldl')
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Axes (CanSelectAxes)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM, backendCoordinates, backendShape, shape, strides)
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
index (Tensor acc) = TensorElement $ acc A.! A.fromIndex sourceShape (A.constant linear)
  where
    sourceShape = A.constant (backendShape (Proxy @sh))
    linear =
      foldl'
        (\offset (dim, coord) -> offset * dim + coord)
        0
        (zip (shape @sh) (coordinates @ix))

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

-- | One source-axis mapping.
-- Fixed axes consume no output coordinate; retained axes transform an output coordinate by an offset and a signed step.
data SliceMapping = SliceMapping
  { sourceDimension :: Int,
    sourceStart :: Int,
    sourceStep :: Int,
    outputAxis :: Maybe (Int, Int)
  }

-- | Prepare the coordinate mapping once from validated slice descriptors.
-- Output axes contain their dimension and row-major stride.
planSlice :: [SliceAxis Int] -> [Int] -> [(Int, Int)] -> [SliceMapping]
planSlice [] [] [] = []
planSlice (At coord : rest) (dim : dims) output =
  SliceMapping dim coord 0 Nothing : planSlice rest dims output
planSlice (selection : rest) (dim : dims) (axis : output) =
  let (start, step) = case selection of
        All -> (0, 1)
        Range begin _ stride -> (begin, stride)
        RangeReverse begin _ stride ->
          (begin + max 0 (fst axis - 1) * stride, negate stride)
   in SliceMapping dim start step (Just axis) : planSlice rest dims output
planSlice _ _ _ = error "Synapse.Tensor.slice: inconsistent selection rank"

-- | Convert an output linear index to a source linear index using the prepared mapping.
-- The fold follows source axis order, including axes removed by At.
applySlicePlan :: A.Exp Int -> [SliceMapping] -> A.Exp Int
applySlicePlan linear = foldl' addAxis 0
  where
    addAxis offset mapping =
      let coord = case outputAxis mapping of
            Nothing -> A.constant (sourceStart mapping)
            Just (dim, stride) ->
              A.constant (sourceStart mapping)
                + A.constant (sourceStep mapping)
                  * ((linear `quot` A.constant stride) `rem` A.constant dim)
       in offset * A.constant (sourceDimension mapping) + coord

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
    sourceShape = A.constant (backendShape (Proxy @sh))
    targetShape = A.constant (backendShape (Proxy @(SliceShape spec sh)))
    targetDims = shape @(SliceShape spec sh)
    targetAxes = zip targetDims (strides targetDims)
    mappings = planSlice (sliceAxes @spec) (shape @sh) targetAxes

    sourceIndex :: A.Exp (ShapeToDIM (SliceShape spec sh)) -> A.Exp (ShapeToDIM sh)
    sourceIndex ix = A.fromIndex sourceShape $ applySlicePlan (A.toIndex targetShape ix) mappings
