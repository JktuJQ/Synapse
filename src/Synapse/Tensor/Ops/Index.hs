{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Static element indexing and multidimensional region selection.
module Synapse.Tensor.Ops.Index (Indexable (..), coordinate, index, unsafeDynamicIndex, dynamicIndexOr, unsafeGather, gatherOr, unsafeLookupRows, slice) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat)
import qualified Synapse.Autograd.Tensor.Ops.Primitives.Index as Pullback
import Synapse.Tensor.Ops (TensorOps (..))
import qualified Synapse.Tensor.Ops.Primitives.Index as Raw
import Synapse.Tensor.Shapes.Axes (CanSelectAxes)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Index (CanGather, CanIndex, CanSlice, Index, LookupRowsShape, Slice, SliceShape)
import Synapse.Tensor.Type (Element, Tensor, TensorElement, TensorIndex)
import Prelude hiding ((!!))

-- | Read a coordinate of a complete index.
-- Axes are numbered from zero, from left to right.
-- The selected axis must exist in the index shape.
coordinate :: forall axis sh. (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) => TensorIndex sh -> TensorElement Int
coordinate = Raw.coordinate @axis

-- | Read one element as an expression, without executing the tensor.
-- For example, @index \@'[1, 2] matrix@.
-- Use scalar to materialize the expression.
index :: forall {t} ix sh a. (TensorOps t a, Shape sh, Index ix, CanIndex ix sh) => t sh a -> Element t a
index = liftElement (Raw.index @ix) (Pullback.index (Proxy @ix))

-- | Read an element at a complete computed index without checking coordinate values.
-- Every coordinate must be nonnegative and strictly less than its dimension.
-- Invalid coordinates have backend-dependent behavior and may cause invalid memory access.
-- Indices carry no differentiation history; gradients propagate only to the selected tensor element.
unsafeDynamicIndex :: (TensorOps t a, Shape sh) => t sh a -> TensorIndex sh -> Element t a
unsafeDynamicIndex tensor selected = liftElement (Raw.indexAt selected) (Pullback.indexAt selected) tensor

-- | Read a computed index, returning the supplied host value when any coordinate is out of bounds.
-- Empty tensors always return the fallback; scalars accept their unique empty index.
-- Invalid selections contribute no gradient to the source tensor.
dynamicIndexOr :: (TensorOps t a, Shape sh) => a -> t sh a -> TensorIndex sh -> Element t a
dynamicIndexOr fallback tensor selected = liftElement (Raw.dynamicIndexOr fallback selected) (Pullback.dynamicIndexOr selected) tensor

-- | Gather along a zero-based axis using an ordinary integer tensor of computed indices.
-- Indices and input have equal ranks and equal dimensions outside the selected axis.
-- The output has the shape of the indices; no broadcasting or negative-index wrapping is performed.
-- Index values must lie in @[0, axisSize)@; invalid values may cause invalid memory access.
-- Gradients accumulate at repeated selections; indices are not differentiated.
unsafeGather :: forall {t} axis sh indices a. (TensorOps t a, KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices) => t sh a -> Tensor indices Int -> t indices a
unsafeGather tensor indices = lift1 (\input -> Raw.unsafeGather @axis input indices) (Pullback.gather @axis False indices) tensor

-- | Gather along an axis, substituting a host fallback for each invalid index.
-- Shape rules and gradient accumulation match 'unsafeGather'.
-- Empty source axes produce only fallback values, and invalid selections contribute no source gradient.
gatherOr :: forall {t} axis sh indices a. (TensorOps t a, KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices) => a -> t sh a -> Tensor indices Int -> t indices a
gatherOr fallback tensor indices = lift1 (\input -> Raw.gatherOr @axis fallback input indices) (Pullback.gather @axis True indices) tensor

-- | Select complete rows of a matrix using an ordinary integer tensor of indices.
-- Appends the matrix column count to the index shape; scalar indices select one row vector.
-- Indices must lie in @[0, rows)@; values are unchecked and invalid indices may cause invalid memory access.
-- Repeated selections accumulate gradients into their source rows; indices are not differentiated.
unsafeLookupRows :: forall {t} rows width indices a. (TensorOps t a, Shape '[rows, width], Shape indices, Shape (LookupRowsShape indices width)) => t '[rows, width] a -> Tensor indices Int -> t (LookupRowsShape indices width) a
unsafeLookupRows table indices = lift1 (`Raw.unsafeLookupRows` indices) (Pullback.lookupRows indices) table

-- | Select a region using one descriptor per axis.
-- At removes an axis; ranges preserve it.
-- Ranges have exclusive stops and strict bounds.
-- A forward range whose stop precedes its start is empty.
-- RangeReverse reverses the coordinates selected by the corresponding Range, without changing the output shape.
slice :: forall {t} spec sh a. (TensorOps t a, Shape sh, Shape (SliceShape spec sh), Slice spec, CanSlice spec sh) => t sh a -> t (SliceShape spec sh) a
slice = lift1 (Raw.slice @spec) (Pullback.slice (Proxy @spec))

-- | Read a coordinate or tensor element using a shape-checked index.
-- Use @ix !! Proxy \@0@ for a coordinate and @tensor !! Proxy \@'[1, 2]@ for a static element.
-- A 'TensorIndex' of the same shape can also be used directly as @tensor !! ix@, without checking coordinate values.
class Indexable source selector result | source selector -> result where
  -- | Build an element expression without executing the tensor.
  (!!) :: source -> selector -> result

infixl 9 !!

instance (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) => Indexable (TensorIndex sh) (Proxy axis) (TensorElement Int) where
  ix !! _ = coordinate @axis ix

instance (TensorOps t a, Shape sh, Index ix, CanIndex ix sh, result ~ Element t a) => Indexable (t sh a) (Proxy ix) result where
  tensor !! _ = index @ix tensor

instance (TensorOps t a, Shape sh, result ~ Element t a) => Indexable (t sh a) (TensorIndex sh) result where
  tensor !! selected = unsafeDynamicIndex tensor selected
