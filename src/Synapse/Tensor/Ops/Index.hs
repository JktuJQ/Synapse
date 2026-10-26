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
module Synapse.Tensor.Ops.Index (Indexable (..), coordinate, index, slice) where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat)
import qualified Synapse.Autograd.Tensor.Primitives.Index as Pullback
import Synapse.Tensor.Ops (TensorOps (..))
import qualified Synapse.Tensor.Primitives.Index as Raw
import Synapse.Tensor.Shapes.Axes (CanSelectAxes)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Index (CanIndex, CanSlice, Index, Slice, SliceShape)
import Synapse.Tensor.Type (Element, TensorElement, TensorIndex)
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

-- | Select a region using one descriptor per axis.
-- At removes an axis; ranges preserve it.
-- Ranges have exclusive stops and strict bounds.
-- A forward range whose stop precedes its start is empty.
-- RangeReverse reverses the coordinates selected by the corresponding Range, without changing the output shape.
slice :: forall {t} spec sh a. (TensorOps t a, Shape sh, Shape (SliceShape spec sh), Slice spec, CanSlice spec sh) => t sh a -> t (SliceShape spec sh) a
slice = lift1 (Raw.slice @spec) (Pullback.slice (Proxy @spec))

-- | Read a coordinate or tensor element using a shape-checked index.
-- Use @ix !! Proxy \@0@ for a coordinate and @tensor !! Proxy \@'[1, 2]@ for a static element.
-- A 'TensorIndex' of the same shape can also be used directly as @tensor !! ix@.
class Indexable source selector result | source selector -> result where
  -- | Build an element expression without executing the tensor.
  (!!) :: source -> selector -> result

infixl 9 !!

instance (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh) => Indexable (TensorIndex sh) (Proxy axis) (TensorElement Int) where
  ix !! _ = coordinate @axis ix

instance (TensorOps t a, Shape sh, Index ix, CanIndex ix sh, result ~ Element t a) => Indexable (t sh a) (Proxy ix) result where
  tensor !! _ = index @ix tensor

instance (TensorOps t a, Shape sh, result ~ Element t a) => Indexable (t sh a) (TensorIndex sh) result where
  tensor !! selected = liftElement (Raw.indexAt selected) (Pullback.indexAt selected) tensor
