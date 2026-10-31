{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Element access and region selection on ordinary staged tensors.
module Synapse.Tensor.Primitives.Index (coordinate, index, indexAt, dynamicIndexOr, unsafeGather, gatherOr, indexInBounds, gatherIndex, gatherReducedIndex, unsafeGatherReduced, slice) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.Shapes.Axes (CanSelectAxes, RemoveAxis)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM, backendCoordinates, backendIndex, backendShape, shape)
import Synapse.Tensor.Shapes.Index
  ( CanGather,
    CanIndex,
    CanSlice,
    Index,
    Slice,
    SliceAxis (..),
    SliceShape,
    coordinates,
    sliceAxes,
  )
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..), TensorIndex (..))
import Prelude hiding ((!!))
import qualified Prelude as P

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

-- | Read one coordinate of a complete index expression.
coordinate :: forall axis sh. (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) => TensorIndex sh -> TensorElement Int
coordinate (TensorIndex ix) = TensorElement (backendCoordinates @sh ix P.!! axisNumber)
  where
    axisNumber = fromInteger (toInteger (natVal (Proxy @axis)))

-- | Read an element at a complete index expression.
indexAt :: (Shape sh, DType a) => TensorIndex sh -> Tensor sh a -> TensorElement a
indexAt (TensorIndex ix) (Tensor acc) = TensorElement (acc A.! ix)

-- | Test every coordinate against its corresponding dimension.
indexInBounds :: forall sh. (Shape sh) => TensorIndex sh -> A.Exp Bool
indexInBounds (TensorIndex ix) =
  foldr (A.&&) (A.constant True) $
    zipWith (\coord dim -> coord A.>= 0 A.&& coord A.< A.constant dim) (backendCoordinates @sh ix) (shape @sh)

-- | Read a complete dynamic index, replacing invalid coordinates with a host fallback.
-- The actual memory access uses clamped coordinates even when the requested index is invalid.
dynamicIndexOr :: forall sh a. (Shape sh, DType a) => a -> TensorIndex sh -> Tensor sh a -> TensorElement a
dynamicIndexOr fallback selected@(TensorIndex ix) (Tensor input)
  | 0 `elem` shape @sh = TensorElement (A.constant fallback)
  | otherwise = TensorElement $ A.cond (indexInBounds selected) (input A.! clamped) (A.constant fallback)
  where
    clamped = backendIndex @sh $ zipWith (\coord dim -> A.max 0 (A.min (A.constant (dim - 1)) coord)) (backendCoordinates @sh ix) (shape @sh)

-- | Replace one coordinate with the corresponding element of an index tensor.
gatherIndex :: forall axis sh indices. (KnownNat axis, Shape sh, Shape indices) => Tensor indices Int -> A.Exp (ShapeToDIM indices) -> TensorIndex sh
gatherIndex (Tensor indices) ix =
  TensorIndex $
    backendIndex @sh $
      zipWith (\position coord -> if position == axis then indices A.! ix else coord) [0 ..] (backendCoordinates @indices ix)
  where
    axis :: Int
    axis = fromInteger (toInteger (natVal (Proxy @axis)))

-- | Gather elements along an axis without checking index values.
unsafeGather :: forall axis sh indices a. (KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, DType a) => Tensor sh a -> Tensor indices Int -> Tensor indices a
unsafeGather input indices = Tensor $ A.generate (A.constant (backendShape (Proxy @indices))) $ \ix ->
  unTensorElement (indexAt (gatherIndex @axis @sh indices ix) input)

-- | Gather elements with a host fallback for out-of-bounds indices.
gatherOr :: forall axis sh indices a. (KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, DType a) => a -> Tensor sh a -> Tensor indices Int -> Tensor indices a
gatherOr fallback input indices = Tensor $ A.generate (A.constant (backendShape (Proxy @indices))) $ \ix ->
  unTensorElement (dynamicIndexOr fallback (gatherIndex @axis @sh indices ix) input)

-- | Read an element at statically checked coordinates.
index :: forall ix sh a. (Shape sh, DType a, Index ix, CanIndex ix sh) => Tensor sh a -> TensorElement a
index = indexAt (TensorIndex (backendIndex @sh (map A.constant (coordinates @ix))))

-- | Backpermute a region into its statically computed output shape.
slice :: forall spec sh a. (Shape sh, Shape (SliceShape spec sh), DType a, Slice spec, CanSlice spec sh) => Tensor sh a -> Tensor (SliceShape spec sh) a
slice (Tensor acc) = Tensor $ A.backpermute targetShape sourceIndex acc
  where
    targetShape = A.constant (backendShape (Proxy @(SliceShape spec sh)))
    targetDims = shape @(SliceShape spec sh)
    sourceIndex :: A.Exp (ShapeToDIM (SliceShape spec sh)) -> A.Exp (ShapeToDIM sh)
    sourceIndex ix = backendIndex @sh (sliceCoordinates (sliceAxes @spec) targetDims (backendCoordinates @(SliceShape spec sh) ix))

-- | Insert the class selected for each coordinate outside the class axis.
gatherReducedIndex :: forall axis sh. (KnownNat axis, Shape sh, Shape (RemoveAxis axis sh)) => Tensor (RemoveAxis axis sh) Int -> A.Exp (ShapeToDIM (RemoveAxis axis sh)) -> TensorIndex sh
gatherReducedIndex (Tensor indices) ix =
  TensorIndex $
    backendIndex @sh $
      take axis coords ++ [indices A.! ix] ++ drop axis coords
  where
    axis :: Int
    axis = fromInteger (toInteger (natVal (Proxy @axis)))
    coords = backendCoordinates @(RemoveAxis axis sh) ix

-- | Select one unchecked class coordinate per observation without constructing one-hot targets.
unsafeGatherReduced :: forall axis sh a. (KnownNat axis, Shape sh, Shape (RemoveAxis axis sh), DType a) => Tensor sh a -> Tensor (RemoveAxis axis sh) Int -> Tensor (RemoveAxis axis sh) a
unsafeGatherReduced input indices = Tensor $ A.generate (A.constant (backendShape (Proxy @(RemoveAxis axis sh)))) $ \ix ->
  unTensorElement (indexAt (gatherReducedIndex @axis @sh indices ix) input)
