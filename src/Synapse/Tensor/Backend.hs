{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Tensor execution backends.
--
-- The public 'run' function is the boundary between staged tensor graphs and
-- materialized tensor values.
module Synapse.Tensor.Backend
  ( Backend (..),
    TensorValue (..),
    run,
    runCPU,
    runGPU,
    toList,
  )
where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
#ifdef SYNAPSE_GPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
#endif

import GHC.TypeLits (Nat)
import Synapse.Tensor.Core (Tensor (..))
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shape (KnownShape, ShapeToDIM)

-- | Execution backend.
data Backend
  = CPU
  | GPU
  deriving (Eq, Show)

-- | A materialized tensor result produced by a backend.
newtype TensorValue (sh :: [Nat]) a
  = TensorValue {unTensorValue :: A.Array (ShapeToDIM sh) a}

-- | Execute a tensor graph and materialize the result.
run :: forall sh a. (KnownShape sh, DType a) => Backend -> Tensor sh a -> TensorValue sh a
run backend tensor =
  case backend of
    CPU -> runCPU tensor
    GPU -> runGPU tensor

-- | Execute a tensor graph on the CPU backend.
runCPU :: (KnownShape sh, DType a) => Tensor sh a -> TensorValue sh a
runCPU = TensorValue . CPU.run . unTensor

-- | Execute a tensor graph on the GPU backend.
runGPU :: forall sh a. (KnownShape sh, DType a) => Tensor sh a -> TensorValue sh a
#ifdef SYNAPSE_GPU
runGPU = TensorValue . GPU.run . unTensor
#else
runGPU _ =
  error "Synapse.Tensor.run: GPU backend is not enabled in this build"
#endif

-- | Convert a materialized tensor result to a row-major list.
toList :: (KnownShape sh, DType a) => TensorValue sh a -> [a]
toList = A.toList . unTensorValue
