{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Static element indexing and multidimensional region selection.
module Synapse.Tensor.Ops.Index (index, slice) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM, backendShape, shape)
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
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))

-- | Read one element as an expression, without executing the tensor.
-- For example, @index \@'[1, 2] matrix@. Use scalar to materialize the expression.
index ::
  forall ix sh a.
  (Shape sh, Index ix, CanIndex ix sh, DType a) =>
  Tensor sh a ->
  TensorElement a
index (Tensor acc) = TensorElement $ acc A.! A.fromIndex sourceShape (A.constant linear)
  where
    sourceShape = A.constant (backendShape (Proxy @sh))
    linear =
      foldl
        (\offset (dim, coordinate) -> offset * dim + coordinate)
        0
        (zip (shape @sh) (coordinates @ix))

-- | Select a region using one descriptor per axis. At removes an axis; ranges
-- preserve it. Ranges have exclusive stops and strict bounds. A forward range
-- whose stop precedes its start is empty. RangeReverse reverses the coordinates
-- selected by the corresponding Range, without changing the output shape.
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
    targetAxes = zip targetDims (drop 1 (scanr (*) 1 targetDims))
    mappings = plan (sliceAxes @spec) (shape @sh) targetAxes

    sourceIndex :: A.Exp (ShapeToDIM (SliceShape spec sh)) -> A.Exp (ShapeToDIM sh)
    sourceIndex ix = A.fromIndex sourceShape $ foldl (addAxis (A.toIndex targetShape ix)) 0 mappings

    addAxis linear offset (dim, start, step, outputAxis) =
      let coordinate = case outputAxis of
            Nothing -> A.constant start
            Just (outputDim, stride) ->
              A.constant start + A.constant step * ((linear `quot` A.constant stride) `rem` A.constant outputDim)
       in offset * A.constant dim + coordinate

    plan [] [] [] = []
    plan (At coordinate : rest) (dim : dims) output =
      (dim, coordinate, 0, Nothing) : plan rest dims output
    plan (selection : rest) (dim : dims) (outputAxis : output) =
      let (start, step) = case selection of
            All -> (0, 1)
            Range begin _ stride -> (begin, stride)
            RangeReverse begin _ stride ->
              (begin + max 0 (fst outputAxis - 1) * stride, negate stride)
       in (dim, start, step, Just outputAxis) : plan rest dims output
    plan _ _ _ = error "Synapse.Tensor.slice: inconsistent selection rank"
