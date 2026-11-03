{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Packing tensor products into backend array products.
module Synapse.Tensor.Backend.Structure (BackendTree (..)) where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM)
import Synapse.Tensor.Type (DType)
import qualified Synapse.Tensor.Type as Raw
import qualified Synapse.Tensor.Type as T

-- | Backend array products preserving tensor shapes and element types.
class (A.Arrays (BackendLeaves a)) => BackendTree a where
  type BackendLeaves a
  type MaterialLeaves a
  pack :: a -> A.Acc (BackendLeaves a)
  unpack :: A.Acc (BackendLeaves a) -> a
  upload :: MaterialLeaves a -> BackendLeaves a
  download :: BackendLeaves a -> MaterialLeaves a

instance (Shape sh, DType a) => BackendTree (T.Tensor sh a) where
  type BackendLeaves (T.Tensor sh a) = A.Array (ShapeToDIM sh) a
  type MaterialLeaves (T.Tensor sh a) = T.TensorData sh a
  pack = Raw.unTensor
  unpack = Raw.Tensor
  upload = Raw.unTensorData
  download = Raw.TensorData

instance (BackendTree a, BackendTree b) => BackendTree (a, b) where
  type BackendLeaves (a, b) = (BackendLeaves a, BackendLeaves b)
  type MaterialLeaves (a, b) = (MaterialLeaves a, MaterialLeaves b)
  pack (a, b) = A.lift (pack a, pack b)
  unpack value = let (a, b) = A.unlift value in (unpack a, unpack b)
  upload (a, b) = (upload @a a, upload @b b)
  download (a, b) = (download @a a, download @b b)
