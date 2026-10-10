{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Tensor execution backends.
--
-- The public 'run' function is the boundary between staged tensor graphs and materialized tensor values.
module Synapse.Tensor.Backend
  ( Backend (..),
    TensorData (..),
    run,
    runCPU,
    runGPU,
  )
where

import qualified Data.Array.Accelerate.LLVM.Native as CPU
#ifdef SYNAPSE_GPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
#endif

import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (Tensor (..), TensorData (..))

-- | Execution backend.
data Backend
  = CPU
  | GPU
  deriving (Eq, Show)

-- | Execute a tensor graph and materialize the result.
run :: forall sh a. (Shape sh, DType a) => Backend -> Tensor sh a -> TensorData sh a
run backend tensor =
  case backend of
    CPU -> runCPU tensor
    GPU -> runGPU tensor

-- | Execute a tensor graph on the CPU backend.
runCPU :: (Shape sh, DType a) => Tensor sh a -> TensorData sh a
runCPU = TensorData . CPU.run . unTensor

-- | Execute a tensor graph on the GPU backend.
runGPU :: forall sh a. (Shape sh, DType a) => Tensor sh a -> TensorData sh a
#ifdef SYNAPSE_GPU
runGPU = TensorData . GPU.run . unTensor
#else
runGPU _ =
  error "Synapse.Tensor.run: GPU backend is not enabled in this build"
#endif
