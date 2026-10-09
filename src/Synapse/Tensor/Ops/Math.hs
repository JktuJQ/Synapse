{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Tensor math operations.
module Synapse.Tensor.Ops.Math () where

import Synapse.Tensor.Construction (full)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Elementwise (map, zipWith)
import Synapse.Tensor.Shape (KnownShape)
import Synapse.Tensor.Type (Tensor (..), TensorElement (..))
import Prelude hiding (map, zip, zipWith)

instance (KnownShape sh, DType a, Num a, Num (TensorElement a)) => Num (Tensor sh a) where
  (+) = zipWith (+)
  (-) = zipWith (-)
  (*) = zipWith (*)
  negate = map negate
  abs = map abs
  signum = map signum
  fromInteger = full @sh . fromInteger

instance (KnownShape sh, DType a, Fractional a, Fractional (TensorElement a)) => Fractional (Tensor sh a) where
  (/) = zipWith (/)
  recip = map recip
  fromRational = full @sh . fromRational

instance (KnownShape sh, DType a, Floating a, Floating (TensorElement a)) => Floating (Tensor sh a) where
  pi = full @sh pi
  exp = map exp
  log = map log
  sqrt = map sqrt
  (**) = zipWith (**)
  logBase = zipWith logBase
  sin = map sin
  cos = map cos
  tan = map tan
  asin = map asin
  acos = map acos
  atan = map atan
  sinh = map sinh
  cosh = map cosh
  tanh = map tanh
  asinh = map asinh
  acosh = map acosh
  atanh = map atanh
