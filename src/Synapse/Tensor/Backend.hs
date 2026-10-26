{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Prepared execution of tensors and tensor records.
-- Records derive @Generic@ and define 'WithTensor' once for both differentiation and execution.
module Synapse.Tensor.Backend (Backend (..), Computation, run, run1, run2, run3, run4, run5, run6, run7, run8, run9) where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
#ifdef SYNAPSE_GPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
#endif
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM)
import Synapse.Tensor.Type (DType, Tensor, TensorData, WithTensor)
import qualified Synapse.Tensor.Type as Raw
import qualified Synapse.Tensor.Type as T

-- | Execution backend.
data Backend = CPU | GPU deriving (Eq, Show)

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

-- | Tensor structures accepted by prepared backend computations.
-- Each invocation packs and reconstructs the record without executing its fields separately.
class (Flatten Tensor a, BackendTree (Leaves Tensor a)) => Computation a where
  inputData :: WithTensor T.TensorData a -> BackendLeaves (Leaves Tensor a)
  outputData :: BackendLeaves (Leaves Tensor a) -> WithTensor T.TensorData a

instance
  ( Flatten Tensor a,
    BackendTree (Leaves Tensor a),
    Flatten TensorData (WithTensor T.TensorData a),
    Leaves TensorData (WithTensor T.TensorData a) ~ MaterialLeaves (Leaves Tensor a)
  ) =>
  Computation a
  where
  inputData = upload @(Leaves Tensor a) . flatten @TensorData
  outputData = rebuild @TensorData . download @(Leaves Tensor a)

-- | Execute all tensor leaves in one backend computation and restore the result structure.
run :: forall a. (Computation a) => Backend -> a -> WithTensor TensorData a
run backend value = outputData @a (execute (pack (flatten @Tensor value)))
  where
    execute = case backend of
      CPU -> CPU.runN
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN
#else
      GPU -> error "Synapse.Tensor.run: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of one tensor structure for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run1 ::
  forall a1 result.
  (Computation a1, Computation result) =>
  Backend ->
  (a1 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData result
run1 backend f = outputData @result . execute . inputData @a1
  where
    program x1 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run1: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 2 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run2 ::
  forall a1 a2 result.
  (Computation a1, Computation a2, Computation result) =>
  Backend ->
  (a1 -> a2 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData result
run2 backend f = \x1 x2 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2))
  where
    program x1 x2 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run2: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 3 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run3 ::
  forall a1 a2 a3 result.
  (Computation a1, Computation a2, Computation a3, Computation result) =>
  Backend ->
  (a1 -> a2 -> a3 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData a3 ->
  WithTensor TensorData result
run3 backend f = \x1 x2 x3 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2) (inputData @a3 x3))
  where
    program x1 x2 x3 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2)) (rebuild @Tensor (unpack x3))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run3: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 4 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run4 ::
  forall a1 a2 a3 a4 result.
  (Computation a1, Computation a2, Computation a3, Computation a4, Computation result) =>
  Backend ->
  (a1 -> a2 -> a3 -> a4 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData a3 ->
  WithTensor TensorData a4 ->
  WithTensor TensorData result
run4 backend f = \x1 x2 x3 x4 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2) (inputData @a3 x3) (inputData @a4 x4))
  where
    program x1 x2 x3 x4 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2)) (rebuild @Tensor (unpack x3)) (rebuild @Tensor (unpack x4))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run4: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 5 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run5 ::
  forall a1 a2 a3 a4 a5 result.
  (Computation a1, Computation a2, Computation a3, Computation a4, Computation a5, Computation result) =>
  Backend ->
  (a1 -> a2 -> a3 -> a4 -> a5 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData a3 ->
  WithTensor TensorData a4 ->
  WithTensor TensorData a5 ->
  WithTensor TensorData result
run5 backend f = \x1 x2 x3 x4 x5 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2) (inputData @a3 x3) (inputData @a4 x4) (inputData @a5 x5))
  where
    program x1 x2 x3 x4 x5 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2)) (rebuild @Tensor (unpack x3)) (rebuild @Tensor (unpack x4)) (rebuild @Tensor (unpack x5))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run5: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 6 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run6 ::
  forall a1 a2 a3 a4 a5 a6 result.
  (Computation a1, Computation a2, Computation a3, Computation a4, Computation a5, Computation a6, Computation result) =>
  Backend ->
  (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData a3 ->
  WithTensor TensorData a4 ->
  WithTensor TensorData a5 ->
  WithTensor TensorData a6 ->
  WithTensor TensorData result
run6 backend f = \x1 x2 x3 x4 x5 x6 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2) (inputData @a3 x3) (inputData @a4 x4) (inputData @a5 x5) (inputData @a6 x6))
  where
    program x1 x2 x3 x4 x5 x6 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2)) (rebuild @Tensor (unpack x3)) (rebuild @Tensor (unpack x4)) (rebuild @Tensor (unpack x5)) (rebuild @Tensor (unpack x6))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run6: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 7 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run7 ::
  forall a1 a2 a3 a4 a5 a6 a7 result.
  (Computation a1, Computation a2, Computation a3, Computation a4, Computation a5, Computation a6, Computation a7, Computation result) =>
  Backend ->
  (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> a7 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData a3 ->
  WithTensor TensorData a4 ->
  WithTensor TensorData a5 ->
  WithTensor TensorData a6 ->
  WithTensor TensorData a7 ->
  WithTensor TensorData result
run7 backend f = \x1 x2 x3 x4 x5 x6 x7 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2) (inputData @a3 x3) (inputData @a4 x4) (inputData @a5 x5) (inputData @a6 x6) (inputData @a7 x7))
  where
    program x1 x2 x3 x4 x5 x6 x7 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2)) (rebuild @Tensor (unpack x3)) (rebuild @Tensor (unpack x4)) (rebuild @Tensor (unpack x5)) (rebuild @Tensor (unpack x6)) (rebuild @Tensor (unpack x7))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run7: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 8 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run8 ::
  forall a1 a2 a3 a4 a5 a6 a7 a8 result.
  (Computation a1, Computation a2, Computation a3, Computation a4, Computation a5, Computation a6, Computation a7, Computation a8, Computation result) =>
  Backend ->
  (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> a7 -> a8 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData a3 ->
  WithTensor TensorData a4 ->
  WithTensor TensorData a5 ->
  WithTensor TensorData a6 ->
  WithTensor TensorData a7 ->
  WithTensor TensorData a8 ->
  WithTensor TensorData result
run8 backend f = \x1 x2 x3 x4 x5 x6 x7 x8 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2) (inputData @a3 x3) (inputData @a4 x4) (inputData @a5 x5) (inputData @a6 x6) (inputData @a7 x7) (inputData @a8 x8))
  where
    program x1 x2 x3 x4 x5 x6 x7 x8 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2)) (rebuild @Tensor (unpack x3)) (rebuild @Tensor (unpack x4)) (rebuild @Tensor (unpack x5)) (rebuild @Tensor (unpack x6)) (rebuild @Tensor (unpack x7)) (rebuild @Tensor (unpack x8))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run8: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 9 tensor structures for repeated execution.
-- Bind the returned function once and pass changing tensors through its inputs.
-- Captured values remain fixed and preparation may compile lazily on the first invocation.
run9 ::
  forall a1 a2 a3 a4 a5 a6 a7 a8 a9 result.
  (Computation a1, Computation a2, Computation a3, Computation a4, Computation a5, Computation a6, Computation a7, Computation a8, Computation a9, Computation result) =>
  Backend ->
  (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> a7 -> a8 -> a9 -> result) ->
  WithTensor TensorData a1 ->
  WithTensor TensorData a2 ->
  WithTensor TensorData a3 ->
  WithTensor TensorData a4 ->
  WithTensor TensorData a5 ->
  WithTensor TensorData a6 ->
  WithTensor TensorData a7 ->
  WithTensor TensorData a8 ->
  WithTensor TensorData a9 ->
  WithTensor TensorData result
run9 backend f = \x1 x2 x3 x4 x5 x6 x7 x8 x9 -> outputData @result (execute (inputData @a1 x1) (inputData @a2 x2) (inputData @a3 x3) (inputData @a4 x4) (inputData @a5 x5) (inputData @a6 x6) (inputData @a7 x7) (inputData @a8 x8) (inputData @a9 x9))
  where
    program x1 x2 x3 x4 x5 x6 x7 x8 x9 = pack (flatten @Tensor (f (rebuild @Tensor (unpack x1)) (rebuild @Tensor (unpack x2)) (rebuild @Tensor (unpack x3)) (rebuild @Tensor (unpack x4)) (rebuild @Tensor (unpack x5)) (rebuild @Tensor (unpack x6)) (rebuild @Tensor (unpack x7)) (rebuild @Tensor (unpack x8)) (rebuild @Tensor (unpack x9))))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run9: GPU backend is not enabled in this build"
#endif
