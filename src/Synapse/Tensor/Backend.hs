{-# LANGUAGE CPP #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Tensor execution backends.
--
-- The public 'run' function is the boundary between staged tensor graphs and materialized tensor values.
module Synapse.Tensor.Backend
  ( Backend (..),
    run,
    run1,
    run2,
    run3,
    run4,
    run5,
    run6,
    run7,
    run8,
    run9,
  )
where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
#ifdef SYNAPSE_GPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
#endif

import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM)
import Synapse.Tensor.Type (Tensor (..), TensorData (..))

-- | Execution backend.
data Backend
  = CPU
  | GPU
  deriving (Eq, Show)

-- | A staged tensor or tuple of computations with a corresponding materialized result.
-- Tensor shapes and element types are preserved, including in nested tuples.
class
  (A.Arrays (ComputationTensors staged)) =>
  Computation staged materialized
    | staged -> materialized,
      materialized -> staged
  where
  -- | Backend representation of the collection of tensors returned by a computation.
  type ComputationTensors staged

  -- | Pack staged outputs into one backend computation without executing it.
  toBackend :: staged -> A.Acc (ComputationTensors staged)

  -- | Wrap the outputs of one backend execution in their materialized tensor types.
  fromBackend :: ComputationTensors staged -> materialized

instance (Shape sh, DType a) => Computation (Tensor sh a) (TensorData sh a) where
  type ComputationTensors (Tensor sh a) = A.Array (ShapeToDIM sh) a
  toBackend = unTensor
  fromBackend = TensorData

instance (Computation a v1, Computation b v2) => Computation (a, b) (v1, v2) where
  type ComputationTensors (a, b) = (ComputationTensors a, ComputationTensors b)
  toBackend (x1, x2) = A.lift (toBackend x1, toBackend x2)
  fromBackend (x1, x2) = (fromBackend x1, fromBackend x2)

instance (Computation a v1, Computation b v2, Computation c v3) => Computation (a, b, c) (v1, v2, v3) where
  type ComputationTensors (a, b, c) = (ComputationTensors a, ComputationTensors b, ComputationTensors c)
  toBackend (x1, x2, x3) = A.lift (toBackend x1, toBackend x2, toBackend x3)
  fromBackend (x1, x2, x3) = (fromBackend x1, fromBackend x2, fromBackend x3)

instance (Computation a v1, Computation b v2, Computation c v3, Computation d v4) => Computation (a, b, c, d) (v1, v2, v3, v4) where
  type ComputationTensors (a, b, c, d) = (ComputationTensors a, ComputationTensors b, ComputationTensors c, ComputationTensors d)
  toBackend (x1, x2, x3, x4) = A.lift (toBackend x1, toBackend x2, toBackend x3, toBackend x4)
  fromBackend (x1, x2, x3, x4) = (fromBackend x1, fromBackend x2, fromBackend x3, fromBackend x4)

instance (Computation a v1, Computation b v2, Computation c v3, Computation d v4, Computation e v5) => Computation (a, b, c, d, e) (v1, v2, v3, v4, v5) where
  type ComputationTensors (a, b, c, d, e) = (ComputationTensors a, ComputationTensors b, ComputationTensors c, ComputationTensors d, ComputationTensors e)
  toBackend (x1, x2, x3, x4, x5) = A.lift (toBackend x1, toBackend x2, toBackend x3, toBackend x4, toBackend x5)
  fromBackend (x1, x2, x3, x4, x5) = (fromBackend x1, fromBackend x2, fromBackend x3, fromBackend x4, fromBackend x5)

instance (Computation a v1, Computation b v2, Computation c v3, Computation d v4, Computation e v5, Computation f v6) => Computation (a, b, c, d, e, f) (v1, v2, v3, v4, v5, v6) where
  type ComputationTensors (a, b, c, d, e, f) = (ComputationTensors a, ComputationTensors b, ComputationTensors c, ComputationTensors d, ComputationTensors e, ComputationTensors f)
  toBackend (x1, x2, x3, x4, x5, x6) = A.lift (toBackend x1, toBackend x2, toBackend x3, toBackend x4, toBackend x5, toBackend x6)
  fromBackend (x1, x2, x3, x4, x5, x6) = (fromBackend x1, fromBackend x2, fromBackend x3, fromBackend x4, fromBackend x5, fromBackend x6)

instance (Computation a v1, Computation b v2, Computation c v3, Computation d v4, Computation e v5, Computation f v6, Computation g v7) => Computation (a, b, c, d, e, f, g) (v1, v2, v3, v4, v5, v6, v7) where
  type ComputationTensors (a, b, c, d, e, f, g) = (ComputationTensors a, ComputationTensors b, ComputationTensors c, ComputationTensors d, ComputationTensors e, ComputationTensors f, ComputationTensors g)
  toBackend (x1, x2, x3, x4, x5, x6, x7) = A.lift (toBackend x1, toBackend x2, toBackend x3, toBackend x4, toBackend x5, toBackend x6, toBackend x7)
  fromBackend (x1, x2, x3, x4, x5, x6, x7) = (fromBackend x1, fromBackend x2, fromBackend x3, fromBackend x4, fromBackend x5, fromBackend x6, fromBackend x7)

instance (Computation a v1, Computation b v2, Computation c v3, Computation d v4, Computation e v5, Computation f v6, Computation g v7, Computation h v8) => Computation (a, b, c, d, e, f, g, h) (v1, v2, v3, v4, v5, v6, v7, v8) where
  type ComputationTensors (a, b, c, d, e, f, g, h) = (ComputationTensors a, ComputationTensors b, ComputationTensors c, ComputationTensors d, ComputationTensors e, ComputationTensors f, ComputationTensors g, ComputationTensors h)
  toBackend (x1, x2, x3, x4, x5, x6, x7, x8) = A.lift (toBackend x1, toBackend x2, toBackend x3, toBackend x4, toBackend x5, toBackend x6, toBackend x7, toBackend x8)
  fromBackend (x1, x2, x3, x4, x5, x6, x7, x8) = (fromBackend x1, fromBackend x2, fromBackend x3, fromBackend x4, fromBackend x5, fromBackend x6, fromBackend x7, fromBackend x8)

instance (Computation a v1, Computation b v2, Computation c v3, Computation d v4, Computation e v5, Computation f v6, Computation g v7, Computation h v8, Computation i v9) => Computation (a, b, c, d, e, f, g, h, i) (v1, v2, v3, v4, v5, v6, v7, v8, v9) where
  type ComputationTensors (a, b, c, d, e, f, g, h, i) = (ComputationTensors a, ComputationTensors b, ComputationTensors c, ComputationTensors d, ComputationTensors e, ComputationTensors f, ComputationTensors g, ComputationTensors h, ComputationTensors i)
  toBackend (x1, x2, x3, x4, x5, x6, x7, x8, x9) = A.lift (toBackend x1, toBackend x2, toBackend x3, toBackend x4, toBackend x5, toBackend x6, toBackend x7, toBackend x8, toBackend x9)
  fromBackend (x1, x2, x3, x4, x5, x6, x7, x8, x9) = (fromBackend x1, fromBackend x2, fromBackend x3, fromBackend x4, fromBackend x5, fromBackend x6, fromBackend x7, fromBackend x8, fromBackend x9)

-- | Execute a tensor or tuple of tensor computations and materialize every output.
run :: (Computation staged materialized) => Backend -> staged -> materialized
run backend computation =
  case backend of
    CPU -> fromBackend (CPU.runN (toBackend computation))
#ifdef SYNAPSE_GPU
    GPU -> fromBackend (GPU.runN (toBackend computation))
#else
    GPU -> error "Synapse.Tensor.run: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of one tensor for repeated execution on materialized inputs.
-- The result may be a tensor or a tuple of tensors with independent shapes and element types.
-- Bind the returned function once and reuse it to retain the prepared computation.
-- Preparation is lazy, so the first application may compile the graph.
-- Values captured by the function remain fixed; pass changing data through its input.
--
-- @
-- let evaluate = run1 CPU (\x -> x * 2 + 1)
-- in evaluate inputData
-- @
run1 ::
  (Shape sh1, DType a, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> result) ->
  (TensorData sh1 a -> value)
run1 backend f = fromBackend . execute . unTensorData
  where
    execute = case backend of
      CPU -> CPU.run1 (toBackend . f . Tensor)
#ifdef SYNAPSE_GPU
      GPU -> GPU.run1 (toBackend . f . Tensor)
#else
      GPU -> error "Synapse.Tensor.run1: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of two tensors for repeated execution on materialized inputs.
-- Bind the returned function once and pass changing data through its two arguments.
-- Input shapes and element types may differ; preparation is lazy as for 'run1'.
run2 ::
  (Shape sh1, Shape sh2, DType a, DType b, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> value)
run2 backend f = \x y -> fromBackend (execute (unTensorData x) (unTensorData y))
  where
    program x y = toBackend (f (Tensor x) (Tensor y))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run2: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of three tensors for repeated execution on materialized inputs.
-- Bind the returned function once and pass changing data through its three arguments.
-- Input shapes and element types may differ; preparation is lazy as for 'run1'.
run3 ::
  (Shape sh1, Shape sh2, Shape sh3, DType a, DType b, DType c, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> Tensor sh3 c -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> TensorData sh3 c -> value)
run3 backend f = \x y z -> fromBackend (execute (unTensorData x) (unTensorData y) (unTensorData z))
  where
    program x y z = toBackend (f (Tensor x) (Tensor y) (Tensor z))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run3: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 4 tensors for repeated execution on materialized inputs.
-- Bind the returned function once; input shapes and element types may differ, as for 'run1'.
run4 ::
  (Shape sh1, Shape sh2, Shape sh3, Shape sh4, DType a, DType b, DType c, DType d, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> Tensor sh3 c -> Tensor sh4 d -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> TensorData sh3 c -> TensorData sh4 d -> value)
run4 backend f = \x1 x2 x3 x4 -> fromBackend (execute (unTensorData x1) (unTensorData x2) (unTensorData x3) (unTensorData x4))
  where
    program x1 x2 x3 x4 = toBackend (f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run4: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 5 tensors for repeated execution on materialized inputs.
-- Bind the returned function once; input shapes and element types may differ, as for 'run1'.
run5 ::
  (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, DType a, DType b, DType c, DType d, DType e, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> Tensor sh3 c -> Tensor sh4 d -> Tensor sh5 e -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> TensorData sh3 c -> TensorData sh4 d -> TensorData sh5 e -> value)
run5 backend f = \x1 x2 x3 x4 x5 -> fromBackend (execute (unTensorData x1) (unTensorData x2) (unTensorData x3) (unTensorData x4) (unTensorData x5))
  where
    program x1 x2 x3 x4 x5 = toBackend (f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run5: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 6 tensors for repeated execution on materialized inputs.
-- Bind the returned function once; input shapes and element types may differ, as for 'run1'.
run6 ::
  (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6, DType a, DType b, DType c, DType d, DType e, DType f, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> Tensor sh3 c -> Tensor sh4 d -> Tensor sh5 e -> Tensor sh6 f -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> TensorData sh3 c -> TensorData sh4 d -> TensorData sh5 e -> TensorData sh6 f -> value)
run6 backend f = \x1 x2 x3 x4 x5 x6 -> fromBackend (execute (unTensorData x1) (unTensorData x2) (unTensorData x3) (unTensorData x4) (unTensorData x5) (unTensorData x6))
  where
    program x1 x2 x3 x4 x5 x6 = toBackend (f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run6: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 7 tensors for repeated execution on materialized inputs.
-- Bind the returned function once; input shapes and element types may differ, as for 'run1'.
run7 ::
  (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6, Shape sh7, DType a, DType b, DType c, DType d, DType e, DType f, DType g, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> Tensor sh3 c -> Tensor sh4 d -> Tensor sh5 e -> Tensor sh6 f -> Tensor sh7 g -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> TensorData sh3 c -> TensorData sh4 d -> TensorData sh5 e -> TensorData sh6 f -> TensorData sh7 g -> value)
run7 backend f = \x1 x2 x3 x4 x5 x6 x7 -> fromBackend (execute (unTensorData x1) (unTensorData x2) (unTensorData x3) (unTensorData x4) (unTensorData x5) (unTensorData x6) (unTensorData x7))
  where
    program x1 x2 x3 x4 x5 x6 x7 = toBackend (f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run7: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 8 tensors for repeated execution on materialized inputs.
-- Bind the returned function once; input shapes and element types may differ, as for 'run1'.
run8 ::
  (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6, Shape sh7, Shape sh8, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> Tensor sh3 c -> Tensor sh4 d -> Tensor sh5 e -> Tensor sh6 f -> Tensor sh7 g -> Tensor sh8 h -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> TensorData sh3 c -> TensorData sh4 d -> TensorData sh5 e -> TensorData sh6 f -> TensorData sh7 g -> TensorData sh8 h -> value)
run8 backend f = \x1 x2 x3 x4 x5 x6 x7 x8 -> fromBackend (execute (unTensorData x1) (unTensorData x2) (unTensorData x3) (unTensorData x4) (unTensorData x5) (unTensorData x6) (unTensorData x7) (unTensorData x8))
  where
    program x1 x2 x3 x4 x5 x6 x7 x8 = toBackend (f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) (Tensor x8))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run8: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of 9 tensors for repeated execution on materialized inputs.
-- Bind the returned function once; input shapes and element types may differ, as for 'run1'.
run9 ::
  (Shape sh1, Shape sh2, Shape sh3, Shape sh4, Shape sh5, Shape sh6, Shape sh7, Shape sh8, Shape sh9, DType a, DType b, DType c, DType d, DType e, DType f, DType g, DType h, DType i, Computation result value) =>
  Backend ->
  (Tensor sh1 a -> Tensor sh2 b -> Tensor sh3 c -> Tensor sh4 d -> Tensor sh5 e -> Tensor sh6 f -> Tensor sh7 g -> Tensor sh8 h -> Tensor sh9 i -> result) ->
  (TensorData sh1 a -> TensorData sh2 b -> TensorData sh3 c -> TensorData sh4 d -> TensorData sh5 e -> TensorData sh6 f -> TensorData sh7 g -> TensorData sh8 h -> TensorData sh9 i -> value)
run9 backend f = \x1 x2 x3 x4 x5 x6 x7 x8 x9 -> fromBackend (execute (unTensorData x1) (unTensorData x2) (unTensorData x3) (unTensorData x4) (unTensorData x5) (unTensorData x6) (unTensorData x7) (unTensorData x8) (unTensorData x9))
  where
    program x1 x2 x3 x4 x5 x6 x7 x8 x9 = toBackend (f (Tensor x1) (Tensor x2) (Tensor x3) (Tensor x4) (Tensor x5) (Tensor x6) (Tensor x7) (Tensor x8) (Tensor x9))
    execute = case backend of
      CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
      GPU -> GPU.runN program
#else
      GPU -> error "Synapse.Tensor.run9: GPU backend is not enabled in this build"
#endif
