{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Core tensor type.
module Synapse.Tensor.Type
  ( Tensor (..),
    TensorElement (..),
    TensorData (..),
    constant,
    FromIntegral (..),
  )
where

import qualified Data.Array.Accelerate as A
import GHC.TypeLits (Nat)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shape (ShapeToDIM)
import Prelude hiding (fromIntegral)

-- | A staged tensor expression.
--
-- The constructor and 'unTensor' provide direct access to Accelerate.
-- When using this escape hatch, callers must ensure runtime dimensions match
-- the type-level shape. Subsequent Synapse operations rely on this invariant.
newtype Tensor (sh :: [Nat]) a
  = Tensor {unTensor :: A.Acc (A.Array (ShapeToDIM sh) a)}

-- | An element expression evaluated inside a tensor computation.
newtype TensorElement a
  = TensorElement {unTensorElement :: A.Exp a}

deriving instance (Num (A.Exp a)) => Num (TensorElement a)

deriving instance (Fractional (A.Exp a)) => Fractional (TensorElement a)

deriving instance (Floating (A.Exp a)) => Floating (TensorElement a)

-- | Materialized input or output data.
-- Direct construction requires runtime dimensions to match the type-level shape,
-- just as for 'Tensor'. The underlying array can be converted using Accelerate.
newtype TensorData (sh :: [Nat]) a
  = TensorData {unTensorData :: A.Array (ShapeToDIM sh) a}

-- | Embed a host value in an element expression.
constant :: (DType a) => a -> TensorElement a
constant = TensorElement . A.constant

-- | Conversion of integral expressions to numeric expressions.
class FromIntegral a b where
  -- | Convert an integral expression without evaluating it on the host.
  fromIntegral :: a -> b

instance (A.Integral a, A.FromIntegral a b) => FromIntegral (TensorElement a) (TensorElement b) where
  fromIntegral (TensorElement x) = TensorElement (A.fromIntegral x)
