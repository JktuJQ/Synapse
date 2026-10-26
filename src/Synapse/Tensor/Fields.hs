{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Shared product traversal for tensor representations.
module Synapse.Tensor.Fields (Leaves, Flatten (..)) where

import GHC.Generics (Generic (..), K1 (..), M1 (..), (:*:) (..))
import Synapse.Tensor.Type (TensorLike)

-- | Preserve tensor leaves and collect record fields into binary products.
type family Leaves (t :: TensorLike) a where
  Leaves t (t sh a) = t sh a
  Leaves t a = GenericLeaves t (Rep a)

-- | The tensor leaves of a single-constructor generic representation.
type family GenericLeaves t f where
  GenericLeaves t (K1 i a) = Leaves t a
  GenericLeaves t (M1 i metadata f) = GenericLeaves t f
  GenericLeaves t (f :*: g) = (GenericLeaves t f, GenericLeaves t g)

-- | Convert a tensor structure to and from its binary field product.
class Flatten (t :: TensorLike) a where
  flatten :: a -> Leaves t a
  rebuild :: Leaves t a -> a

instance {-# OVERLAPPING #-} Flatten t (t sh a) where
  flatten = id
  rebuild = id

-- | Convert generic fields without changing tensor values.
class GenericFlatten t f where
  genericFlatten :: f p -> GenericLeaves t f
  genericRebuild :: GenericLeaves t f -> f p

instance (Flatten t a) => GenericFlatten t (K1 i a) where
  genericFlatten (K1 value) = flatten @t value
  genericRebuild = K1 . rebuild @t

instance (GenericFlatten t f) => GenericFlatten t (M1 i metadata f) where
  genericFlatten (M1 value) = genericFlatten @t value
  genericRebuild = M1 . genericRebuild @t

instance (GenericFlatten t f, GenericFlatten t g) => GenericFlatten t (f :*: g) where
  genericFlatten (left :*: right) = (genericFlatten @t left, genericFlatten @t right)
  genericRebuild (left, right) = genericRebuild @t left :*: genericRebuild @t right

instance {-# OVERLAPPABLE #-} (Generic a, GenericFlatten t (Rep a), Leaves t a ~ GenericLeaves t (Rep a)) => Flatten t a where
  flatten = genericFlatten @t . from
  rebuild = to . genericRebuild @t
