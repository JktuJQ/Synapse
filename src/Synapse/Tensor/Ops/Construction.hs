{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Tensor construction from values and coordinate expressions.
module Synapse.Tensor.Ops.Construction
    ( -- * Tensor construction
      fromGen
    , full
    , fill
    , zeros
    , ones
    , eye
    , identity

      -- * Matching existing tensors
    , fullLike
    , zerosLike
    , onesLike

      -- * Sequences
    , arange
    , linspace
    , logspace

      -- * Element expressions
    , element
    , scalar
    ) where

import Synapse.Tensor.Ops.Definition (TensorOps (Element, fromElement, fromTensor))
import Synapse.Tensor.Ops.Elementwise (FromIntegral)
import Synapse.Tensor.Ops.Raw.Construction (element)
import qualified Synapse.Tensor.Ops.Raw.Construction as Raw
import Synapse.Tensor.Representation (TensorElement, TensorIndex, TensorLike)
import Synapse.Tensor.Shapes.Dimensions (Shape)

-- | Generate each element from its multidimensional index.
fromGen
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh)
    => (TensorIndex sh -> TensorElement a)
    -> t sh a
fromGen = fromTensor . Raw.fromGen

-- | Create a tensor filled with the supplied value.
full
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh)
    => a
    -> t sh a
full = fromTensor . Raw.full

-- | Alias for 'full'.
fill
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh)
    => a
    -> t sh a
fill = full @sh

-- | Create a tensor filled with zeros.
zeros
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, Num a)
    => t sh a
zeros = full @sh 0

-- | Create a tensor filled with ones.
ones
    :: forall {t :: TensorLike} sh a
     . (TensorOps t a, Shape sh, Num a)
    => t sh a
ones = full @sh 1

-- | Create a matrix with ones on the main diagonal and zeros elsewhere.
eye
    :: forall {t :: TensorLike} rows cols a
     . (TensorOps t a, Shape '[rows, cols], Num (TensorElement a))
    => t '[rows, cols] a
eye = fromTensor Raw.eye

-- | Create a square identity matrix.
identity
    :: forall {t :: TensorLike} n a
     . (TensorOps t a, Shape '[n, n], Num (TensorElement a))
    => t '[n, n] a
identity = eye @n @n

-- | Create a tensor with the same shape as another tensor, filled with a value.
fullLike
    :: forall {t :: TensorLike} sh a b
     . (TensorOps t a, Shape sh)
    => t sh b
    -> a
    -> t sh a
fullLike _ value = fromTensor (Raw.full @sh value)

-- | Create a zero tensor with the same shape as another tensor.
zerosLike
    :: forall {t :: TensorLike} sh a b
     . (TensorOps t a, Shape sh, Num a)
    => t sh b
    -> t sh a
zerosLike _ = zeros @sh

-- | Create a ones tensor with the same shape as another tensor.
onesLike
    :: forall {t :: TensorLike} sh a b
     . (TensorOps t a, Shape sh, Num a)
    => t sh b
    -> t sh a
onesLike _ = ones @sh

-- | Create a vector with evenly spaced values @start, start + step, ...@.
arange
    :: forall {t :: TensorLike} n a
     . (TensorOps t a, Shape '[n], Num (TensorElement a), FromIntegral (TensorElement Int) (TensorElement a))
    => a
    -> a
    -> t '[n] a
arange x y = fromTensor $ Raw.arange x y

-- | Create a vector with @n@ evenly spaced values from @start@ to @end@.
linspace
    :: forall {t :: TensorLike} n a
     . ( TensorOps t a
       , Shape '[n]
       , Fractional a
       , Fractional (TensorElement a)
       , FromIntegral (TensorElement Int) (TensorElement a)
       )
    => a
    -> a
    -> t '[n] a
linspace x y = fromTensor $ Raw.linspace x y

-- | Exponentiate 'linspace' using the supplied base.
logspace
    :: forall {t :: TensorLike} n a
     . ( TensorOps t a
       , Shape '[n]
       , Floating a
       , Floating (TensorElement a)
       , FromIntegral (TensorElement Int) (TensorElement a)
       )
    => a
    -> a
    -> a
    -> t '[n] a
logspace x y z = fromTensor $ Raw.logspace x y z

-- | Convert an element expression to a rank-zero tensor.
scalar
    :: forall {t :: TensorLike} a
     . TensorOps t a
    => Element t a
    -> t '[] a
scalar = fromElement
