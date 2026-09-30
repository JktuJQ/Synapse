{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Tensor math operations.
module Synapse.Tensor.Ops.Math () where

import qualified Data.Array.Accelerate as A
import Synapse.Tensor.Construction (full)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shape (KnownShape)
import Synapse.Tensor.Type (Tensor (..))
import Prelude
  ( Floating,
    Fractional,
    Num,
    (.),
  )
import qualified Prelude as P

instance (KnownShape sh, DType a, Num a, Num (A.Exp a)) => Num (Tensor sh a) where
  Tensor lhs + Tensor rhs = Tensor (A.zipWith (P.+) lhs rhs)
  Tensor lhs - Tensor rhs = Tensor (A.zipWith (P.-) lhs rhs)
  Tensor lhs * Tensor rhs = Tensor (A.zipWith (P.*) lhs rhs)
  negate (Tensor tensor) = Tensor (A.map P.negate tensor)
  abs (Tensor tensor) = Tensor (A.map P.abs tensor)
  signum (Tensor tensor) = Tensor (A.map P.signum tensor)
  fromInteger = full @sh . P.fromInteger

instance (KnownShape sh, DType a, Fractional a, Fractional (A.Exp a)) => Fractional (Tensor sh a) where
  Tensor lhs / Tensor rhs = Tensor (A.zipWith (P./) lhs rhs)
  recip (Tensor tensor) = Tensor (A.map P.recip tensor)
  fromRational = full @sh . P.fromRational

instance (KnownShape sh, DType a, Floating a, Floating (A.Exp a)) => Floating (Tensor sh a) where
  pi = full @sh P.pi
  exp (Tensor tensor) = Tensor (A.map P.exp tensor)
  log (Tensor tensor) = Tensor (A.map P.log tensor)
  sqrt (Tensor tensor) = Tensor (A.map P.sqrt tensor)
  (Tensor lhs) ** (Tensor rhs) = Tensor (A.zipWith (P.**) lhs rhs)
  logBase (Tensor lhs) (Tensor rhs) = Tensor (A.zipWith P.logBase lhs rhs)
  sin (Tensor tensor) = Tensor (A.map P.sin tensor)
  cos (Tensor tensor) = Tensor (A.map P.cos tensor)
  tan (Tensor tensor) = Tensor (A.map P.tan tensor)
  asin (Tensor tensor) = Tensor (A.map P.asin tensor)
  acos (Tensor tensor) = Tensor (A.map P.acos tensor)
  atan (Tensor tensor) = Tensor (A.map P.atan tensor)
  sinh (Tensor tensor) = Tensor (A.map P.sinh tensor)
  cosh (Tensor tensor) = Tensor (A.map P.cosh tensor)
  tanh (Tensor tensor) = Tensor (A.map P.tanh tensor)
  asinh (Tensor tensor) = Tensor (A.map P.asinh tensor)
  acosh (Tensor tensor) = Tensor (A.map P.acosh tensor)
  atanh (Tensor tensor) = Tensor (A.map P.atanh tensor)
