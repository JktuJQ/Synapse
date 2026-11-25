{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Element access, indexed selection, and slicing.
module Synapse.Tensor.Ops.Index
    ( -- * Coordinates and static element access
      Indexable (..)
    , coordinate
    , index
    , unsafeDynamicIndex
    , dynamicIndexOr
    , slice

      -- * Gather and scatter
    , unsafeGather
    , gatherOr
    , unsafeLookupRows
    , scatterAdd
    , unsafeScatterSet
    ) where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat)
import Synapse.Autograd.Tensor.Ops.Pullback.Index qualified as Pullback
import Synapse.Tensor.Ops.Constraints (NumDType)
import Synapse.Tensor.Ops.Definition (TensorOps (..), withVJP)
import Synapse.Tensor.Ops.Raw.Construction qualified as Construction
import Synapse.Tensor.Ops.Raw.Index qualified as Raw
import Synapse.Tensor.Representation (Tensor, TensorElement, TensorIndex, TensorLike)
import Synapse.Tensor.Shapes.Axes (CanSelectAxes)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Shapes.Index (CanGather, CanIndex, CanSlice, Index, LookupRowsShape, Slice, SliceShape)
import Prelude hiding ((!!))

{- | Read a coordinate or tensor element using a shape-checked index.
Use @ix !! Proxy \@0@ for a coordinate and @tensor !! Proxy \@'[1, 2]@ for a static element.
A t'TensorIndex' of the same shape can also be used directly as @tensor !! ix@;
coordinate values are not checked.
-}
class Indexable source selector result | source selector -> result where
    -- | Build an element expression.
    (!!)
        :: source
        -> selector
        -> result

infixl 9 !!

instance
    (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh)
    => Indexable (TensorIndex sh) (Proxy axis) (TensorElement Int)
    where
    ix !! _ = coordinate @axis ix
instance
    ( TensorOps t a
    , Shape sh
    , Index ix
    , CanIndex ix sh
    , result ~ Element t a
    )
    => Indexable (t sh a) (Proxy ix) result
    where
    tensor !! _ = index @ix tensor
instance
    (TensorOps t a, Shape sh, result ~ Element t a)
    => Indexable (t sh a) (TensorIndex sh) result
    where
    tensor !! selected = unsafeDynamicIndex tensor selected

-- | Read a coordinate of a complete index.
coordinate
    :: forall axis sh
     . (KnownNat axis, Shape sh, CanSelectAxes '[axis] sh)
    => TensorIndex sh
    -> TensorElement Int
coordinate = Raw.coordinate @axis

-- | Read the element at the specified coordinates.
index
    :: forall {t :: TensorLike} ix sh a
     . (TensorOps t a, Shape sh, Index ix, CanIndex ix sh)
    => t sh a
    -> Element t a
index =
    customElementVJP
        (Proxy @(Num (TensorElement a)))
        (Raw.index @ix)
        (\input -> (Raw.index @ix input, input))
        (Pullback.index (Proxy @ix))

-- | Read an element at computed coordinates. Coordinates must be within the tensor bounds.
unsafeDynamicIndex
    :: forall t sh a
     . (TensorOps t a, Shape sh)
    => t sh a
    -> TensorIndex sh
    -> Element t a
unsafeDynamicIndex tensor selected =
    customElementVJP
        (Proxy @(Num (TensorElement a)))
        (Raw.indexAt selected)
        (\input -> (Raw.indexAt selected input, input))
        (Pullback.indexAt selected)
        tensor

-- | Read an element at computed coordinates, returning the fallback for out-of-bounds indices.
dynamicIndexOr
    :: forall t sh a
     . (TensorOps t a, Shape sh)
    => a
    -> t sh a
    -> TensorIndex sh
    -> Element t a
dynamicIndexOr fallback tensor selected =
    customElementVJP
        (Proxy @(Num (TensorElement a)))
        (Raw.dynamicIndexOr fallback selected)
        (\input -> (Raw.dynamicIndexOr fallback selected input, input))
        (Pullback.dynamicIndexOr selected)
        tensor

-- | Select the specified coordinates and ranges along each axis.
slice
    :: forall {t :: TensorLike} spec sh a
     . ( TensorOps t a
       , Shape sh
       , Shape (SliceShape spec sh)
       , Slice spec
       , CanSlice spec sh
       )
    => t sh a
    -> t (SliceShape spec sh) a
slice =
    withVJP
        (Proxy @(Num (TensorElement a)))
        (Raw.slice @spec)
        (Pullback.slice (Proxy @spec))


-- | Select elements along an axis using an index tensor. Indices must be within the source bounds.
unsafeGather
    :: forall {t :: TensorLike} axis sh indices a
     . ( TensorOps t a
       , KnownNat axis
       , Shape sh
       , Shape indices
       , CanGather axis sh indices
       )
    => t sh a
    -> Tensor indices Int
    -> t indices a
unsafeGather tensor indices =
    withVJP
        (Proxy @(Num (TensorElement a)))
        (\input -> Raw.unsafeGather @axis input indices)
        (Pullback.gather @axis False indices)
        tensor

-- | Select elements along an axis, returning the fallback for out-of-bounds indices.
gatherOr
    :: forall {t :: TensorLike} axis sh indices a
     . ( TensorOps t a
       , KnownNat axis
       , Shape sh
       , Shape indices
       , CanGather axis sh indices
       )
    => a
    -> t sh a
    -> Tensor indices Int
    -> t indices a
gatherOr fallback tensor indices =
    withVJP
        (Proxy @(Num (TensorElement a)))
        (\input -> Raw.gatherOr @axis fallback input indices)
        (Pullback.gather @axis True indices)
        tensor

-- | Select table rows using an index tensor. Row indices must be within the table bounds.
unsafeLookupRows
    :: forall {t :: TensorLike} rows width indices a
     . ( TensorOps t a
       , Shape '[rows, width]
       , Shape indices
       , Shape (LookupRowsShape indices width)
       )
    => t '[rows, width] a
    -> Tensor indices Int
    -> t (LookupRowsShape indices width) a
unsafeLookupRows table indices =
    withVJP
        (Proxy @(Num (TensorElement a)))
        (`Raw.unsafeLookupRows` indices)
        (Pullback.lookupRows indices)
        table

-- | Add updates at the given indices along an axis. Out-of-bounds indices are ignored.
scatterAdd
    :: forall {t :: TensorLike} axis sh indices a
     . (TensorOps t a, KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, NumDType a)
    => t sh a
    -> Tensor indices Int
    -> t indices a
    -> t sh a
scatterAdd base indices updates =
    withVJP
        (Proxy @(() :: Constraint))
        (uncurry $ \input -> Raw.scatterWith @axis (+) input indices)
        (\_ _ seed -> (seed, Raw.gatherOr @axis 0 seed indices))
        (base, updates)

{- | Write updates at the given indices along an axis. Out-of-bounds indices are ignored.

Each destination must be written at most once;
if case this is not true, write order is undefined and in backward propagation both writes get a gradient.
-}
unsafeScatterSet
    :: forall {t :: TensorLike} axis sh indices a
     . (TensorOps t a, KnownNat axis, Shape sh, Shape indices, CanGather axis sh indices, Num a)
    => t sh a
    -> Tensor indices Int
    -> t indices a
    -> t sh a
unsafeScatterSet base indices updates =
    withVJP
        (Proxy @(() :: Constraint))
        (uncurry $ \input -> Raw.scatterWith @axis const input indices)
        ( \_ _ seed ->
            (Raw.scatterWith @axis const seed indices (Construction.full 0), Raw.gatherOr @axis 0 seed indices)
        )
        (base, updates)
