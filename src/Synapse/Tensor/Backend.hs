{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Execution of tensor computations and preparation of reusable tensor functions.
module Synapse.Tensor.Backend
    ( -- * Backends
      Backend (..)

      -- * Supported tensor structures
    , BackendTree (..)
    , Computation (..)

      -- * Closed computations
    , run

      -- * Prepared functions
    , run1
    , run2
    , run3
    , run4
    , run5
    , run6
    , run7
    , run8
    , run9
    ) where

import Synapse.Tensor.Representation (DType, Tensor (..), TensorData (..), WithTensor)
import Synapse.Tensor.Shapes.Dimensions (Shape, ShapeToDIM)
import Synapse.Tensor.Structure (Flatten (..), Leaves)

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
#ifdef SYNAPSE_GPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
#endif

{- | Backend used to compile and execute a tensor computation.

'CPU' is always available.
'GPU' requires building Synapse with the Cabal @gpu@ flag and a supported CUDA device.
Selecting 'GPU' in a build without GPU support raises an error when execution is demanded.
-}
data Backend
    = -- | Native CPU execution through LLVM.
      CPU
    | -- | CUDA GPU execution through LLVM and PTX.
      GPU
    deriving stock (Eq, Show)

{- | Convert a tensor or a binary product of tensors to its backend array representation.

'Leaves' decomposes records into tensors and nested pairs before these conversions are applied.
Each tensor becomes an Accelerate array; each pair becomes a pair of backend representations.
Conversions preserve shapes, element types, and contents.

'pack' and 'unpack' must preserve the meaning of the staged computation in either direction.
'toBackend' and 'fromBackend' must be mutual inverses on valid materialized values.
-}
class A.Arrays (BackendLeaves a) => BackendTree a where
    {- | Materialized Accelerate arrays corresponding to a staged tensor product.

    For @Tensor sh a@ this is @A.Array (ShapeToDIM sh) a@.
    For @(left, right)@ it is @(BackendLeaves left, BackendLeaves right)@.
    Wrap this type in 'A.Acc' to represent the corresponding staged computation.
    -}
    type BackendLeaves a

    {- | Materialized Synapse tensors with the same binary product structure.

    A @Tensor sh a@ becomes @TensorData sh a@; pairs are converted recursively.
    Unlike 'BackendLeaves', this representation retains Synapse's shape-indexed wrappers.
    -}
    type MaterialLeaves a

    -- | Combine tensor leaves into an array product.
    pack
        :: a
        -> A.Acc (BackendLeaves a)

    -- | Split an array product into tensor leaves.
    unpack
        :: A.Acc (BackendLeaves a)
        -> a

    -- | Convert tensor data leaves to arrays.
    toBackend
        :: MaterialLeaves a
        -> BackendLeaves a

    -- | Convert arrays to tensor data leaves.
    fromBackend
        :: BackendLeaves a
        -> MaterialLeaves a

instance (Shape sh, DType a) => BackendTree (Tensor sh a) where
    type BackendLeaves (Tensor sh a) = A.Array (ShapeToDIM sh) a
    type MaterialLeaves (Tensor sh a) = TensorData sh a
    pack = unTensor
    unpack = Tensor
    toBackend = unTensorData
    fromBackend = TensorData
instance (BackendTree a, BackendTree b) => BackendTree (a, b) where
    type BackendLeaves (a, b) = (BackendLeaves a, BackendLeaves b)
    type MaterialLeaves (a, b) = (MaterialLeaves a, MaterialLeaves b)
    pack (a, b) = A.lift (pack a, pack b)
    unpack value = let (a, b) = A.unlift value in (unpack a, unpack b)
    toBackend (a, b) = (toBackend @a a, toBackend @b b)
    fromBackend (a, b) = (fromBackend @a a, fromBackend @b b)

{- | Tensor structures that can be executed as one backend computation.

Instances are inferred for tensors and supported tensor records; no explicit instance is needed.
'WithTensor' replaces each staged tensor with its materialized counterpart.
The structure may contain tensors with different shapes and element types.
-}
class (Flatten Tensor a, BackendTree (Leaves Tensor a)) => Computation a where
    -- | Convert a tensor data structure to its array product.
    inputData
        :: WithTensor TensorData a
        -> BackendLeaves (Leaves Tensor a)

    -- | Reconstruct a tensor data structure from its array product.
    outputData
        :: BackendLeaves (Leaves Tensor a)
        -> WithTensor TensorData a

instance
    ( Flatten Tensor a
    , BackendTree (Leaves Tensor a)
    , Flatten TensorData (WithTensor TensorData a)
    , Leaves TensorData (WithTensor TensorData a) ~ MaterialLeaves (Leaves Tensor a)
    )
    => Computation a
    where
    inputData = toBackend @(Leaves Tensor a) . flatten @TensorData
    outputData = rebuild @TensorData . fromBackend @(Leaves Tensor a)

-- | Evaluate a tensor computation and return its tensor data.
run
    :: forall a
     . Computation a
    => Backend
    -> a
    -> WithTensor TensorData a
run backend = outputData @a . execute . pack . flatten @Tensor
  where
    execute = case backend of
        CPU -> CPU.runN
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN
#else
        GPU -> error "Synapse.Tensor.run: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of one tensor argument for repeated execution.
run1
    :: forall a1 result
     . (Computation a1, Computation result)
    => Backend
    -> (a1 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData result
run1 backend f = outputData @result . execute . inputData @a1
  where
    program x1 = pack $ flatten @Tensor $ f $ rebuild @Tensor $ unpack x1
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run1: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of two tensor arguments for repeated execution.
run2
    :: forall a1 a2 result
     . (Computation a1, Computation a2, Computation result)
    => Backend
    -> (a1 -> a2 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData result
run2 backend f = \x1 x2 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
  where
    program x1 x2 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run2: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of three tensor arguments for repeated execution.
run3
    :: forall a1 a2 a3 result
     . (Computation a1, Computation a2, Computation a3, Computation result)
    => Backend
    -> (a1 -> a2 -> a3 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData a3
    -> WithTensor TensorData result
run3 backend f = \x1 x2 x3 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
            (inputData @a3 x3)
  where
    program x1 x2 x3 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
                    (rebuild @Tensor $ unpack x3)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run3: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of four tensor arguments for repeated execution.
run4
    :: forall a1 a2 a3 a4 result
     . (Computation a1, Computation a2, Computation a3, Computation a4, Computation result)
    => Backend
    -> (a1 -> a2 -> a3 -> a4 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData a3
    -> WithTensor TensorData a4
    -> WithTensor TensorData result
run4 backend f = \x1 x2 x3 x4 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
            (inputData @a3 x3)
            (inputData @a4 x4)
  where
    program x1 x2 x3 x4 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
                    (rebuild @Tensor $ unpack x3)
                    (rebuild @Tensor $ unpack x4)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run4: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of five tensor arguments for repeated execution.
run5
    :: forall a1 a2 a3 a4 a5 result
     . (Computation a1, Computation a2, Computation a3, Computation a4, Computation a5, Computation result)
    => Backend
    -> (a1 -> a2 -> a3 -> a4 -> a5 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData a3
    -> WithTensor TensorData a4
    -> WithTensor TensorData a5
    -> WithTensor TensorData result
run5 backend f = \x1 x2 x3 x4 x5 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
            (inputData @a3 x3)
            (inputData @a4 x4)
            (inputData @a5 x5)
  where
    program x1 x2 x3 x4 x5 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
                    (rebuild @Tensor $ unpack x3)
                    (rebuild @Tensor $ unpack x4)
                    (rebuild @Tensor $ unpack x5)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run5: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of six tensor arguments for repeated execution.
run6
    :: forall a1 a2 a3 a4 a5 a6 result
     . ( Computation a1
       , Computation a2
       , Computation a3
       , Computation a4
       , Computation a5
       , Computation a6
       , Computation result
       )
    => Backend
    -> (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData a3
    -> WithTensor TensorData a4
    -> WithTensor TensorData a5
    -> WithTensor TensorData a6
    -> WithTensor TensorData result
run6 backend f = \x1 x2 x3 x4 x5 x6 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
            (inputData @a3 x3)
            (inputData @a4 x4)
            (inputData @a5 x5)
            (inputData @a6 x6)
  where
    program x1 x2 x3 x4 x5 x6 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
                    (rebuild @Tensor $ unpack x3)
                    (rebuild @Tensor $ unpack x4)
                    (rebuild @Tensor $ unpack x5)
                    (rebuild @Tensor $ unpack x6)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run6: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of seven tensor arguments for repeated execution.
run7
    :: forall a1 a2 a3 a4 a5 a6 a7 result
     . ( Computation a1
       , Computation a2
       , Computation a3
       , Computation a4
       , Computation a5
       , Computation a6
       , Computation a7
       , Computation result
       )
    => Backend
    -> (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> a7 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData a3
    -> WithTensor TensorData a4
    -> WithTensor TensorData a5
    -> WithTensor TensorData a6
    -> WithTensor TensorData a7
    -> WithTensor TensorData result
run7 backend f = \x1 x2 x3 x4 x5 x6 x7 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
            (inputData @a3 x3)
            (inputData @a4 x4)
            (inputData @a5 x5)
            (inputData @a6 x6)
            (inputData @a7 x7)
  where
    program x1 x2 x3 x4 x5 x6 x7 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
                    (rebuild @Tensor $ unpack x3)
                    (rebuild @Tensor $ unpack x4)
                    (rebuild @Tensor $ unpack x5)
                    (rebuild @Tensor $ unpack x6)
                    (rebuild @Tensor $ unpack x7)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run7: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of eight tensor arguments for repeated execution.
run8
    :: forall a1 a2 a3 a4 a5 a6 a7 a8 result
     . ( Computation a1
       , Computation a2
       , Computation a3
       , Computation a4
       , Computation a5
       , Computation a6
       , Computation a7
       , Computation a8
       , Computation result
       )
    => Backend
    -> (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> a7 -> a8 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData a3
    -> WithTensor TensorData a4
    -> WithTensor TensorData a5
    -> WithTensor TensorData a6
    -> WithTensor TensorData a7
    -> WithTensor TensorData a8
    -> WithTensor TensorData result
run8 backend f = \x1 x2 x3 x4 x5 x6 x7 x8 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
            (inputData @a3 x3)
            (inputData @a4 x4)
            (inputData @a5 x5)
            (inputData @a6 x6)
            (inputData @a7 x7)
            (inputData @a8 x8)
  where
    program x1 x2 x3 x4 x5 x6 x7 x8 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
                    (rebuild @Tensor $ unpack x3)
                    (rebuild @Tensor $ unpack x4)
                    (rebuild @Tensor $ unpack x5)
                    (rebuild @Tensor $ unpack x6)
                    (rebuild @Tensor $ unpack x7)
                    (rebuild @Tensor $ unpack x8)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run8: GPU backend is not enabled in this build"
#endif

-- | Prepare a function of nine tensor arguments for repeated execution.
run9
    :: forall a1 a2 a3 a4 a5 a6 a7 a8 a9 result
     . ( Computation a1
       , Computation a2
       , Computation a3
       , Computation a4
       , Computation a5
       , Computation a6
       , Computation a7
       , Computation a8
       , Computation a9
       , Computation result
       )
    => Backend
    -> (a1 -> a2 -> a3 -> a4 -> a5 -> a6 -> a7 -> a8 -> a9 -> result)
    -> WithTensor TensorData a1
    -> WithTensor TensorData a2
    -> WithTensor TensorData a3
    -> WithTensor TensorData a4
    -> WithTensor TensorData a5
    -> WithTensor TensorData a6
    -> WithTensor TensorData a7
    -> WithTensor TensorData a8
    -> WithTensor TensorData a9
    -> WithTensor TensorData result
run9 backend f = \x1 x2 x3 x4 x5 x6 x7 x8 x9 ->
    outputData @result $
        execute
            (inputData @a1 x1)
            (inputData @a2 x2)
            (inputData @a3 x3)
            (inputData @a4 x4)
            (inputData @a5 x5)
            (inputData @a6 x6)
            (inputData @a7 x7)
            (inputData @a8 x8)
            (inputData @a9 x9)
  where
    program x1 x2 x3 x4 x5 x6 x7 x8 x9 =
        pack $
            flatten @Tensor $
                f
                    (rebuild @Tensor $ unpack x1)
                    (rebuild @Tensor $ unpack x2)
                    (rebuild @Tensor $ unpack x3)
                    (rebuild @Tensor $ unpack x4)
                    (rebuild @Tensor $ unpack x5)
                    (rebuild @Tensor $ unpack x6)
                    (rebuild @Tensor $ unpack x7)
                    (rebuild @Tensor $ unpack x8)
                    (rebuild @Tensor $ unpack x9)
    execute = case backend of
        CPU -> CPU.runN program
#ifdef SYNAPSE_GPU
        GPU -> GPU.runN program
#else
        GPU -> error "Synapse.Tensor.run9: GPU backend is not enabled in this build"
#endif
