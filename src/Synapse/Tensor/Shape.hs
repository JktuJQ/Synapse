{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Type-level tensor shapes.
--
-- Synapse represents tensor shapes as type-level lists of natural numbers. This
-- module provides runtime helpers for those type-level shapes.
module Synapse.Tensor.Shape
  ( ShapeToDIM,
    KnownShape (..),
    shape,
    shapeSize,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import GHC.TypeLits
  ( ErrorMessage (ShowType, Text, (:<>:)),
    KnownNat,
    Nat,
    TypeError,
    natVal,
  )

-- | Convert a type-level Synapse shape to the internal backend shape type.
type family ShapeToDIM (sh :: [Nat]) :: Type where
  ShapeToDIM '[] = A.DIM0
  ShapeToDIM '[d1] = A.DIM1
  ShapeToDIM '[d1, d2] = A.DIM2
  ShapeToDIM '[d1, d2, d3] = A.DIM3
  ShapeToDIM '[d1, d2, d3, d4] = A.DIM4
  ShapeToDIM '[d1, d2, d3, d4, d5] = A.DIM5
  ShapeToDIM '[d1, d2, d3, d4, d5, d6] = A.DIM6
  ShapeToDIM '[d1, d2, d3, d4, d5, d6, d7] = A.DIM7
  ShapeToDIM '[d1, d2, d3, d4, d5, d6, d7, d8] = A.DIM8
  ShapeToDIM '[d1, d2, d3, d4, d5, d6, d7, d8, d9] = A.DIM9
  ShapeToDIM sh =
    TypeError
      ( 'Text "Synapse tensors currently support rank 0..9, but got shape: "
          ':<>: 'ShowType sh
      )

-- | Runtime representation of a type-level tensor shape.
class (A.Shape (ShapeToDIM sh)) => KnownShape (sh :: [Nat]) where
  -- | Convert the type-level shape to the internal backend shape value.
  shapeVal :: proxy sh -> ShapeToDIM sh

  -- | Convert the type-level shape to a list of runtime dimensions.
  shapeList :: proxy sh -> [Int]

instance KnownShape '[] where
  shapeVal _ = A.Z
  shapeList _ = []

instance (KnownNat d1) => KnownShape '[d1] where
  shapeVal _ = A.Z A.:. dimVal @d1
  shapeList _ = [dimVal @d1]

instance (KnownNat d1, KnownNat d2) => KnownShape '[d1, d2] where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2
  shapeList _ = [dimVal @d1, dimVal @d2]

instance (KnownNat d1, KnownNat d2, KnownNat d3) => KnownShape '[d1, d2, d3] where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2 A.:. dimVal @d3
  shapeList _ = [dimVal @d1, dimVal @d2, dimVal @d3]

instance
  (KnownNat d1, KnownNat d2, KnownNat d3, KnownNat d4) =>
  KnownShape '[d1, d2, d3, d4]
  where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2 A.:. dimVal @d3 A.:. dimVal @d4
  shapeList _ = [dimVal @d1, dimVal @d2, dimVal @d3, dimVal @d4]

instance
  (KnownNat d1, KnownNat d2, KnownNat d3, KnownNat d4, KnownNat d5) =>
  KnownShape '[d1, d2, d3, d4, d5]
  where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2 A.:. dimVal @d3 A.:. dimVal @d4 A.:. dimVal @d5
  shapeList _ = [dimVal @d1, dimVal @d2, dimVal @d3, dimVal @d4, dimVal @d5]

instance
  (KnownNat d1, KnownNat d2, KnownNat d3, KnownNat d4, KnownNat d5, KnownNat d6) =>
  KnownShape '[d1, d2, d3, d4, d5, d6]
  where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2 A.:. dimVal @d3 A.:. dimVal @d4 A.:. dimVal @d5 A.:. dimVal @d6
  shapeList _ = [dimVal @d1, dimVal @d2, dimVal @d3, dimVal @d4, dimVal @d5, dimVal @d6]

instance
  (KnownNat d1, KnownNat d2, KnownNat d3, KnownNat d4, KnownNat d5, KnownNat d6, KnownNat d7) =>
  KnownShape '[d1, d2, d3, d4, d5, d6, d7]
  where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2 A.:. dimVal @d3 A.:. dimVal @d4 A.:. dimVal @d5 A.:. dimVal @d6 A.:. dimVal @d7
  shapeList _ = [dimVal @d1, dimVal @d2, dimVal @d3, dimVal @d4, dimVal @d5, dimVal @d6, dimVal @d7]

instance
  (KnownNat d1, KnownNat d2, KnownNat d3, KnownNat d4, KnownNat d5, KnownNat d6, KnownNat d7, KnownNat d8) =>
  KnownShape '[d1, d2, d3, d4, d5, d6, d7, d8]
  where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2 A.:. dimVal @d3 A.:. dimVal @d4 A.:. dimVal @d5 A.:. dimVal @d6 A.:. dimVal @d7 A.:. dimVal @d8
  shapeList _ = [dimVal @d1, dimVal @d2, dimVal @d3, dimVal @d4, dimVal @d5, dimVal @d6, dimVal @d7, dimVal @d8]

instance
  (KnownNat d1, KnownNat d2, KnownNat d3, KnownNat d4, KnownNat d5, KnownNat d6, KnownNat d7, KnownNat d8, KnownNat d9) =>
  KnownShape '[d1, d2, d3, d4, d5, d6, d7, d8, d9]
  where
  shapeVal _ = A.Z A.:. dimVal @d1 A.:. dimVal @d2 A.:. dimVal @d3 A.:. dimVal @d4 A.:. dimVal @d5 A.:. dimVal @d6 A.:. dimVal @d7 A.:. dimVal @d8 A.:. dimVal @d9
  shapeList _ = [dimVal @d1, dimVal @d2, dimVal @d3, dimVal @d4, dimVal @d5, dimVal @d6, dimVal @d7, dimVal @d8, dimVal @d9]

-- | Runtime dimensions for a type-level shape.
shape :: forall sh. (KnownShape sh) => [Int]
shape = shapeList (Proxy @sh)

-- | Total number of elements in a type-level shape.
shapeSize :: forall sh. (KnownShape sh) => Int
shapeSize = product (shape @sh)

-- | Convert a type-level natural number to an 'Int'.
dimVal :: forall n. (KnownNat n) => Int
dimVal = fromInteger . toInteger $ natVal (Proxy @n)
