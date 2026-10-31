{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Product decomposition and named traversal of tensor fields.
module Synapse.Tensor.Fields (Leaves, Flatten (..), FieldPath, TensorFields (..)) where

import Data.Int (Int16, Int32, Int64, Int8)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Typeable (Typeable)
import Data.Word (Word16, Word32, Word64, Word8)
import GHC.Generics (C, D, Generic (..), K1 (..), M1 (..), S, Selector (..), U1 (..), (:*:) (..))
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (DType, TensorData, TensorLike)

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

-- | Nested record selector names, or zero-based positions for unnamed fields.
-- A standalone tensor uses the empty path.
type FieldPath = [String]

-- | Visit materialized tensor fields while preserving ordinary fields.
-- Generic instances support single-constructor records, newtypes, and tuples.
-- Ordinary numeric values, booleans, and characters are left unchanged.
-- Lists and optional fields retain their length and constructor during traversal.
-- Deriving 'Generic' is sufficient when every field has a 'TensorFields' instance.
-- Custom instances can select tensor fields, assign stable paths, or preserve opaque configuration.
-- A configuration type with no tensor fields may implement @traverseTensors _ = pure@.
-- Arbitrary sum types require a custom instance; traversal preserves the active constructor.
--
-- @
-- instance T.TensorFields Custom where
--   traverseTensors visit (Custom tensor configuration) =
--     (\updated -> Custom updated configuration) <$> visit ["weights"] tensor
-- @
class TensorFields p where
  -- | Transform each tensor using its path and static shape evidence.
  -- Instances must assign unique paths and preserve the surrounding structure.
  -- The callback order and paths must not depend on tensor contents.
  -- Visiting with @\_ tensor -> pure tensor@ must preserve the value.
  -- Prefix nested paths when delegating to another 'traverseTensors' call to avoid name collisions.
  traverseTensors ::
    (Applicative f) =>
    (forall sh a. (Shape sh, DType a, Typeable a) => FieldPath -> TensorData sh a -> f (TensorData sh a)) ->
    p ->
    f p

instance {-# OVERLAPPING #-} (Shape sh, DType a, Typeable a) => TensorFields (TensorData sh a) where
  traverseTensors visit = visit []

instance TensorFields Double where traverseTensors _ = pure

instance TensorFields Float where traverseTensors _ = pure

instance TensorFields Int where traverseTensors _ = pure

instance TensorFields Int8 where traverseTensors _ = pure

instance TensorFields Int16 where traverseTensors _ = pure

instance TensorFields Int32 where traverseTensors _ = pure

instance TensorFields Int64 where traverseTensors _ = pure

instance TensorFields Integer where traverseTensors _ = pure

instance TensorFields Word where traverseTensors _ = pure

instance TensorFields Word8 where traverseTensors _ = pure

instance TensorFields Word16 where traverseTensors _ = pure

instance TensorFields Word32 where traverseTensors _ = pure

instance TensorFields Word64 where traverseTensors _ = pure

instance TensorFields Bool where traverseTensors _ = pure

instance TensorFields Char where traverseTensors _ = pure

-- | Preserve text without visiting its characters.
instance {-# OVERLAPPING #-} TensorFields String where
  traverseTensors _ = pure

instance (TensorFields a) => TensorFields [a] where
  traverseTensors visit values = traverse (\(index, value) -> traverseTensors (\path -> visit (show index : path)) value) (zip [0 :: Int ..] values)

instance (TensorFields a) => TensorFields (Maybe a) where
  traverseTensors _ Nothing = pure Nothing
  traverseTensors visit (Just value) = Just <$> traverseTensors (\path -> visit ("Just" : path)) value

-- | Generic traversal of product fields with positional names for unnamed selectors.
class GenericTensorFields (f :: Type -> Type) where
  fieldCount :: proxy f -> Int
  genericTensors ::
    (Applicative m) =>
    (forall sh a. (Shape sh, DType a, Typeable a) => FieldPath -> TensorData sh a -> m (TensorData sh a)) ->
    Int ->
    f x ->
    m (f x)

instance (TensorFields a) => GenericTensorFields (K1 i a) where
  fieldCount _ = 1
  genericTensors visit _ (K1 value) = K1 <$> traverseTensors visit value

instance (GenericTensorFields f, GenericTensorFields g) => GenericTensorFields (f :*: g) where
  fieldCount _ = fieldCount (Proxy @f) + fieldCount (Proxy @g)
  genericTensors visit offset (left :*: right) = (:*:) <$> genericTensors visit offset left <*> genericTensors visit (offset + fieldCount (Proxy @f)) right

instance (GenericTensorFields f) => GenericTensorFields (M1 D metadata f) where
  fieldCount _ = fieldCount (Proxy @f)
  genericTensors visit offset (M1 value) = M1 <$> genericTensors visit offset value

instance (GenericTensorFields f) => GenericTensorFields (M1 C metadata f) where
  fieldCount _ = fieldCount (Proxy @f)
  genericTensors visit offset (M1 value) = M1 <$> genericTensors visit offset value

instance (Selector metadata, GenericTensorFields f) => GenericTensorFields (M1 S metadata f) where
  fieldCount _ = 1
  genericTensors visit offset field@(M1 value) =
    let name = case selName field of
          "" -> show offset
          selector -> selector
     in M1 <$> genericTensors (\path -> visit (name : path)) 0 value

instance GenericTensorFields U1 where
  fieldCount _ = 0
  genericTensors _ _ U1 = pure U1

instance {-# OVERLAPPABLE #-} (Generic p, GenericTensorFields (Rep p)) => TensorFields p where
  traverseTensors visit value = to <$> genericTensors visit 0 (from value)
