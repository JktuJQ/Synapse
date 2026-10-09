{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Explicit tensor operations on zero-based axes counted from left to right.
module Synapse.Tensor.Ops.Axes
  ( squeeze,
    unsqueeze,
    permuteAxes,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Axes
  ( Axes,
    CanPermuteAxes,
    CanSqueeze,
    InsertAxis,
    PermuteAxes,
    RemoveAxis,
    axes,
  )
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM, backendShape, shape)
import Synapse.Tensor.Type (Tensor (..))

-- Private: the public axis operations establish preservation of element count.
reshapeAxes ::
  forall to from a.
  (Shape from, Shape to, DType a) =>
  Tensor from a ->
  Tensor to a
reshapeAxes (Tensor acc) = Tensor $ A.backpermute targetShape sourceIndex acc
  where
    sourceShape = A.constant (backendShape (Proxy @from))
    targetShape = A.constant (backendShape (Proxy @to))
    sourceIndex ix = A.fromIndex sourceShape (A.toIndex targetShape ix)

-- | Remove one explicitly selected singleton axis.
squeeze ::
  forall axis sh a.
  (Shape sh, Shape (RemoveAxis axis sh), CanSqueeze axis sh, DType a) =>
  Tensor sh a ->
  Tensor (RemoveAxis axis sh) a
squeeze = reshapeAxes @(RemoveAxis axis sh)

-- | Insert a singleton axis, including before the first or after the last axis.
unsqueeze ::
  forall axis sh a.
  (Shape sh, Shape (InsertAxis axis 1 sh), DType a) =>
  Tensor sh a ->
  Tensor (InsertAxis axis 1 sh) a
unsqueeze = reshapeAxes @(InsertAxis axis 1 sh)

-- | Reorder axes. Output axis i corresponds to the source axis xs[i].
-- Every source axis must appear once; this changes element order as well as shape.
permuteAxes ::
  forall xs sh a.
  (Shape sh, Shape (PermuteAxes xs sh), Axes xs, CanPermuteAxes xs sh, DType a) =>
  Tensor sh a ->
  Tensor (PermuteAxes xs sh) a
permuteAxes (Tensor acc) = Tensor $ A.backpermute targetShape sourceIndex acc
  where
    sourceShape = A.constant (backendShape (Proxy @sh))
    targetShape = A.constant (backendShape (Proxy @(PermuteAxes xs sh)))
    sourceStrides = strides (shape @sh)
    targetDims = shape @(PermuteAxes xs sh)
    targetStrides = strides targetDims
    mappings = zip3 targetDims targetStrides [sourceStrides !! axis | axis <- axes @xs]

    sourceIndex :: A.Exp (ShapeToDIM (PermuteAxes xs sh)) -> A.Exp (ShapeToDIM sh)
    sourceIndex ix = A.fromIndex sourceShape $ foldr (addAxis (A.toIndex targetShape ix)) 0 mappings

    addAxis linear (dim, targetStride, sourceStride) total =
      let coordinate = (linear `quot` A.constant targetStride) `rem` A.constant dim
       in coordinate * A.constant sourceStride + total

    strides dims = drop 1 (scanr (*) 1 dims)
