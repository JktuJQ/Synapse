{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}

{- | Indexing and slicing. - Static geometry is validated by shape constraints; unchecked dynamic reads rely
on caller-supplied bounds.
-}
module Synapse.Tensor.Ops.Raw.Index
    ( -- * Coordinates and static element access
      coordinate
    , index
    , indexAt
    , indexInBounds
    , dynamicIndexOr
    , slice

      -- * Gather and scatter
    , gatherIndex
    , unsafeGather
    , gatherOr
    , gatherReducedIndex
    , unsafeGatherReduced
    , unsafeLookupRows
    , scatterWith
    ) where

import Data.Array.Accelerate qualified as A
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, natVal)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..), TensorIndex (..))
import Synapse.Tensor.Shapes.Axes (CanSelectAxes, RemoveAxis)
import Synapse.Tensor.Shapes.Backend (ShapeToDIM, backendCoordinates, backendIndex, backendShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shape)
import Synapse.Tensor.Shapes.Index
    ( CanGather
    , CanIndex
    , CanSlice
    , Index
    , LookupRowsShape
    , Slice
    , SliceAxis (..)
    , SliceShape
    , coordinates
    , sliceAxes
    )
import Prelude hiding ((!!))
import Prelude qualified as P

-- | Map retained coordinates directly to source coordinates, inserting fixed coordinates for removed axes.
sliceCoordinates
    :: [SliceAxis Int]
    -> [Int]
    -> [A.Exp Int]
    -> [A.Exp Int]
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
coordinate
    :: forall axis sh
     . (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh)
    => TensorIndex sh
    -> TensorElement Int
coordinate (TensorIndex ix) = TensorElement (backendCoordinates @sh ix P.!! axisNumber)
  where
    axisNumber = fromInteger (toInteger (natVal (Proxy @axis)))

-- | Read the element at the specified coordinates.
index
    :: forall ix sh a
     . (Shape sh, DType a, Index ix, CanIndex ix sh)
    => Tensor sh a
    -> TensorElement a
index = indexAt (TensorIndex (backendIndex @sh (map A.constant (coordinates @ix))))

-- | Read an element at a complete index expression.
indexAt
    :: (Shape sh, DType a)
    => TensorIndex sh
    -> Tensor sh a
    -> TensorElement a
indexAt (TensorIndex ix) (Tensor acc) = TensorElement (acc A.! ix)

-- | Test every coordinate against its corresponding dimension.
indexInBounds
    :: forall sh
     . Shape sh
    => TensorIndex sh
    -> A.Exp Bool
indexInBounds (TensorIndex ix) =
    foldr (A.&&) (A.constant True) $
        zipWith (\coord dim -> coord A.>= 0 A.&& coord A.< A.constant dim) (backendCoordinates @sh ix) (shape @sh)

-- | Read an element at computed coordinates, returning the fallback for out-of-bounds indices.
dynamicIndexOr
    :: forall sh a
     . (Shape sh, DType a)
    => a
    -> TensorIndex sh
    -> Tensor sh a
    -> TensorElement a
dynamicIndexOr fallback selected@(TensorIndex ix) (Tensor input)
    | 0 `elem` shape @sh = TensorElement (A.constant fallback)
    | otherwise = TensorElement $ A.cond (indexInBounds selected) (input A.! clamped) (A.constant fallback)
  where
    clamped =
        backendIndex @sh $
            zipWith (\coord dim -> A.max 0 (A.min (A.constant (dim - 1)) coord)) (backendCoordinates @sh ix) (shape @sh)

-- | Select the specified coordinates and ranges along each axis.
slice
    :: forall spec sh a
     . (Shape sh, Shape (SliceShape spec sh), DType a, Slice spec, CanSlice spec sh)
    => Tensor sh a
    -> Tensor (SliceShape spec sh) a
slice (Tensor acc) = Tensor $ A.backpermute targetShape sourceIndex acc
  where
    targetShape = A.constant (backendShape (Proxy @(SliceShape spec sh)))
    targetDims = shape @(SliceShape spec sh)

    sourceIndex ix =
        backendIndex @sh (sliceCoordinates (sliceAxes @spec) targetDims (backendCoordinates @(SliceShape spec sh) ix))

-- | Replace one coordinate with the corresponding element of an index tensor.
gatherIndex
    :: forall axis sh indices
     . (KnownNat axis, Shape sh, Shape indices)
    => Tensor indices Int
    -> A.Exp (ShapeToDIM indices)
    -> TensorIndex sh
gatherIndex (Tensor indices) ix =
    TensorIndex $
        backendIndex @sh $
            zipWith
                (\position coord -> if position == axis then indices A.! ix else coord)
                [0 ..]
                (backendCoordinates @indices ix)
  where
    axis = fromIntegral @_ @Int (natVal (Proxy @axis))

-- | Select elements along an axis using an index tensor. Indices must be within the source bounds.
unsafeGather
    :: forall axis sh indices a
     . (KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, DType a)
    => Tensor sh a
    -> Tensor indices Int
    -> Tensor indices a
unsafeGather input indices = Tensor $ A.generate (A.constant (backendShape (Proxy @indices))) $ \ix ->
    unTensorElement (indexAt (gatherIndex @axis @sh indices ix) input)

-- | Select elements along an axis, returning the fallback for out-of-bounds indices.
gatherOr
    :: forall axis sh indices a
     . (KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, DType a)
    => a
    -> Tensor sh a
    -> Tensor indices Int
    -> Tensor indices a
gatherOr fallback input indices = Tensor $ A.generate (A.constant (backendShape (Proxy @indices))) $ \ix ->
    unTensorElement (dynamicIndexOr fallback (gatherIndex @axis @sh indices ix) input)

-- | Insert the class selected for each coordinate outside the class axis.
gatherReducedIndex
    :: forall axis sh
     . (KnownNat axis, Shape sh, Shape (RemoveAxis axis sh))
    => Tensor (RemoveAxis axis sh) Int
    -> A.Exp (ShapeToDIM (RemoveAxis axis sh))
    -> TensorIndex sh
gatherReducedIndex (Tensor indices) ix =
    TensorIndex $
        backendIndex @sh $
            take axis coords ++ [indices A.! ix] ++ drop axis coords
  where
    axis = fromInteger (toInteger (natVal (Proxy @axis)))
    coords = backendCoordinates @(RemoveAxis axis sh) ix

-- | Select one unchecked class coordinate per observation without constructing one-hot targets.
unsafeGatherReduced
    :: forall axis sh a
     . (KnownNat axis, Shape sh, Shape (RemoveAxis axis sh), DType a)
    => Tensor sh a
    -> Tensor (RemoveAxis axis sh) Int
    -> Tensor (RemoveAxis axis sh) a
unsafeGatherReduced input indices = Tensor $ A.generate (A.constant (backendShape (Proxy @(RemoveAxis axis sh)))) $ \ix ->
    unTensorElement (indexAt (gatherReducedIndex @axis @sh indices ix) input)

-- | Select table rows using an index tensor. Row indices must be within the table bounds.
unsafeLookupRows
    :: forall rows width indices a
     . (Shape indices, Shape (LookupRowsShape indices width), DType a)
    => Tensor '[rows, width] a
    -> Tensor indices Int
    -> Tensor (LookupRowsShape indices width) a
unsafeLookupRows (Tensor table) (Tensor indices) =
    let outputShape = A.constant $ backendShape $ Proxy @(LookupRowsShape indices width)
        columns = A.indexHead $ A.shape table
        selected = A.flatten indices
     in Tensor $ A.generate outputShape $ \ix ->
            let offset = A.toIndex outputShape ix
                row = selected A.! A.index1 (offset `A.quot` columns)
             in table A.! A.index2 row (offset `A.rem` columns)

-- | Combine updates with destination elements along an axis, discarding out-of-bounds indices.
scatterWith
    :: forall axis sh indices a
     . (KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, DType a)
    => (TensorElement a -> TensorElement a -> TensorElement a)
    -> Tensor sh a
    -> Tensor indices Int
    -> Tensor indices a
    -> Tensor sh a
scatterWith combine (Tensor base) indices (Tensor updates) =
    Tensor $ A.permute merge base destination updates
  where
    merge new old = unTensorElement $ combine (TensorElement new) (TensorElement old)
    destination ix =
        let selected@(TensorIndex coord) = gatherIndex @axis @sh indices ix
         in A.cond (indexInBounds selected) (A.Just_ coord) A.Nothing_
