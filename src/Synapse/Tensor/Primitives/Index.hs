{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Element access and region selection on ordinary staged tensors.
module Synapse.Tensor.Primitives.Index (coordinate, index, indexAt, slice) where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
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
