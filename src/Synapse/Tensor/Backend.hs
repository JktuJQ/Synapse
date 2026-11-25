{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskellQuotes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

{- | Execute staged tensor computations on the CPU or a CUDA device.

Choose 'defaultCPU' or 'defaultGPU' for the backend's default target.
To select a worker count or device explicitly, create a target once and wrap it in 'CPU' or 'GPU'.
Reuse that backend to prepare as many computations as needed.

'run' accepts a closed tensor expression or a function with any number of arguments.
Each argument and the result may be a single tensor, a tuple, or a record with a 'TensorStructure' instance.
The prepared function takes and returns the same structures with 'TensorData' in place of 'Tensor'.
-}
module Synapse.Tensor.Backend
    ( -- * Targets
      Backend (..)
    , CPUTarget
    , GPUTarget
    , defaultCPU
    , defaultGPU
    , createCPUTarget
    , createGPUTarget

      -- * Execution
    , RunnableTensor
    , Runnable
    , RunResult
    , run
    , run'

      -- * Asynchronous execution
    , Async
    , RunAsyncResult
    , runAsync
    , runAsync'
    , waitAsync
    , pollAsync
    , cancelAsync
    ) where

import Control.Applicative ((<|>))
import Control.Monad (unless)
import Data.Proxy (Proxy (..))
import GHC.Conc (getNumCapabilities)

import Data.Array.Accelerate.LLVM.Native qualified as Native
#ifdef SYNAPSE_GPU
import Control.Concurrent (runInBoundThread)
import Data.Array.Accelerate.LLVM.PTX qualified as PTX
import Foreign.CUDA.Driver qualified as CUDA
#endif

import Synapse.Tensor.Representation (DType, Tensor (..), TensorData (..))
import Synapse.Tensor.Shapes.Backend (ShapeToDIM)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Structure
    ( TensorField (..)
    , TensorProduct
    , TensorStructure
    , WithTensor
    , flatten
    , rebuild
    )

import Data.Array.Accelerate qualified as A
import Language.Haskell.TH qualified as TH
import Language.Haskell.TH.Syntax qualified as TH
import Text.Read (readMaybe)
import Type.Reflection qualified as Reflection

-- | Select an execution target for 'run'.
data Backend
    = CPU CPUTarget
    | GPU GPUTarget
    deriving stock (Show)

-- | A CPU worker pool shared by computations prepared with the same target.
data CPUTarget = DefaultCPU | CPUWorkers Int Native.Native

instance Show CPUTarget where
    showsPrec _ DefaultCPU = showString "default"
    showsPrec _ (CPUWorkers workers _) = showString "workers=" . shows workers

-- | The default CUDA device pool or a context on one explicitly selected device.
data GPUTarget
    = DefaultGPU
#ifdef SYNAPSE_GPU
    | GPUDevice Int PTX.PTX
#endif

instance Show GPUTarget where
    showsPrec _ DefaultGPU = showString "default"
#ifdef SYNAPSE_GPU
    showsPrec _ (GPUDevice device _) = showString "device=" . shows device
#endif

{- | Use the default CPU worker pool.

Set @ACCELERATE_LLVM_NATIVE_THREADS@ before the first CPU execution to override its worker count.
-}
defaultCPU :: Backend
defaultCPU = CPU DefaultCPU

{- | Use the default pool of visible CUDA devices.

To restrict the pool, set @ACCELERATE_LLVM_PTX_DEVICES@ before the first GPU execution.
Use a list of device ordinals such as @[0]@.
-}
defaultGPU :: Backend
defaultGPU = GPU DefaultGPU

{- | Create a pool with one worker per capability, starting at capability zero.

The worker count must be positive and be at most the number of RTS capabilities.
-}
createCPUTarget
    :: Int
    -> IO CPUTarget
createCPUTarget workers = do
    capabilities <- getNumCapabilities
    unless (workers > 0 && workers <= capabilities) $
        fail "createCPUTarget: worker count must be between 1 and the number of RTS capabilities"
    CPUWorkers workers <$> Native.createTarget [0 .. workers - 1]

{- | Create a CUDA context for a device visible to the process, numbered from zero.

Fails if the device is unavailable or Synapse was built without the Cabal @gpu@ flag.
-}
createGPUTarget
    :: Int
    -> IO GPUTarget
#ifdef SYNAPSE_GPU
createGPUTarget ordinal = runInBoundThread $ do
    CUDA.initialise []
    count <- CUDA.count
    unless (ordinal >= 0 && ordinal < count) $
        fail "createGPUTarget: CUDA device is not visible to this process"
    device <- CUDA.device ordinal
    properties <- CUDA.props device
    GPUDevice ordinal <$> PTX.createTargetForDevice device properties [CUDA.SchedAuto]
#else
createGPUTarget _ = fail "createGPUTarget: Synapse was built without GPU support"
#endif

-- | Prepare an Accelerate program using the selected pool or context.
execute
    :: Native.Afunction f
    => Backend
    -> f
    -> Native.AfunctionR f
execute (CPU DefaultCPU) = Native.runN
execute (CPU (CPUWorkers _ target)) = Native.runNWith target
#ifdef SYNAPSE_GPU
execute (GPU DefaultGPU) = PTX.runN
execute (GPU (GPUDevice _ target)) = PTX.runNWith target
#else
execute (GPU _) = error "run: Synapse was built without GPU support"
#endif

{- | Convert a tensor or a binary product of tensors to its backend array representation.

'flatten' decomposes records into named tensor fields and nested pairs.
Each tensor becomes an Accelerate array; each pair becomes a pair of backend representations.
Conversions preserve shapes, element types, and contents.

'pack' and 'unpack' must preserve the meaning of the staged computation in either direction.
-}
class A.Arrays (BackendLeaves a) => BackendTree a where
    {- | Materialized Accelerate arrays corresponding to a staged tensor product.

    For @TensorField (Tensor sh a)@ this is @A.Array (ShapeToDIM sh) a@.
    For @(left, right)@ it is @(BackendLeaves left, BackendLeaves right)@.
    Wrap this type in 'A.Acc' to represent the corresponding staged computation.
    -}
    type BackendLeaves a

    {- | Materialized Synapse tensors with the same binary product structure.

    A @TensorField (Tensor sh a)@ becomes @TensorField (TensorData sh a)@; pairs are converted recursively.
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

instance (Shape sh, DType a) => BackendTree (TensorField (Tensor sh a)) where
    type BackendLeaves (TensorField (Tensor sh a)) = A.Array (ShapeToDIM sh) a
    type MaterialLeaves (TensorField (Tensor sh a)) = TensorField (TensorData sh a)
    pack = unTensor . tensorFieldValue
    unpack = TensorField [] . Tensor
    toBackend = unTensorData . tensorFieldValue
    fromBackend = TensorField [] . TensorData
instance (BackendTree a, BackendTree b) => BackendTree (a, b) where
    type BackendLeaves (a, b) = (BackendLeaves a, BackendLeaves b)
    type MaterialLeaves (a, b) = (MaterialLeaves a, MaterialLeaves b)
    pack (a, b) = A.lift (pack a, pack b)
    unpack value = let (a, b) = A.unlift value in (unpack a, unpack b)
    toBackend (a, b) = (toBackend @a a, toBackend @b b)
    fromBackend (a, b) = (fromBackend @a a, fromBackend @b b)

-- | Translate a closed expression or function to its staged backend type.
type family BackendProgram f where
    BackendProgram (arg -> rest) = A.Acc (BackendLeaves (TensorProduct Tensor arg)) -> BackendProgram rest
    BackendProgram a = A.Acc (BackendLeaves (TensorProduct Tensor a))

-- | Replace each argument and the result with its materialized tensor structure.
type family RunResult f where
    RunResult (arg -> rest) = WithTensor TensorData arg -> RunResult rest
    RunResult a = WithTensor TensorData a

-- Bundle arguments so the public unary Accelerate async entry point suffices.
type family AsyncInputs f where
    AsyncInputs (arg -> rest) = (BackendLeaves (TensorProduct Tensor arg), AsyncInputs rest)
    AsyncInputs a = ()

type family AsyncOutput f where
    AsyncOutput (arg -> rest) = AsyncOutput rest
    AsyncOutput a = BackendLeaves (TensorProduct Tensor a)

-- | The type of the asynchronous runner for a staged program.
type family RunAsyncResult f where
    RunAsyncResult (arg -> rest) = WithTensor TensorData arg -> RunAsyncResult rest
    RunAsyncResult a = IO (Async (WithTensor TensorData a))

-- | A staged tensor structure or a function taking tensor structures and returning one.
class
    (Native.Afunction (BackendProgram f), A.Arrays (AsyncInputs f), A.Arrays (AsyncOutput f)) =>
    Runnable f
    where
    -- | Rebuild arguments from staged arrays and pack the staged result.
    toBackendProgram
        :: f
        -> BackendProgram f

    -- | Adapt a prepared backend function to accept tensor data, or rebuild a closed expression's result.
    fromBackendResult
        :: Proxy f
        -> Native.AfunctionR (BackendProgram f)
        -> RunResult f

    -- | Build the staged program with all inputs in a single array product.
    toBackendAsyncProgram
        :: f
        -> A.Acc (AsyncInputs f)
        -> A.Acc (AsyncOutput f)

    -- | Pack the curried inputs and defer adapting the result until it is requested.
    fromBackendAsyncProgram
        :: Proxy f
        -> (AsyncInputs f -> IO (Native.Async (AsyncOutput f)))
        -> RunAsyncResult f

instance
    ( TensorStructure arg
    , BackendTree (TensorProduct Tensor arg)
    , TensorProduct TensorData arg ~ MaterialLeaves (TensorProduct Tensor arg)
    , Runnable rest
    )
    => Runnable (arg -> rest)
    where
    toBackendProgram f = toBackendProgram . f . rebuild @arg @Tensor . unpack
    fromBackendResult _ f =
        fromBackendResult (Proxy @rest) . f . toBackend @(TensorProduct Tensor arg) . flatten @arg @TensorData
    toBackendAsyncProgram f inputs =
        let (arg, rest) = A.unlift inputs
         in toBackendAsyncProgram (f $ rebuild @arg @Tensor $ unpack arg) rest
    fromBackendAsyncProgram _ f arg =
        fromBackendAsyncProgram (Proxy @rest) $ \rest ->
            f (toBackend @(TensorProduct Tensor arg) (flatten @arg @TensorData arg), rest)
instance
    {-# OVERLAPPABLE #-}
    ( TensorStructure a
    , BackendTree (TensorProduct Tensor a)
    , TensorProduct TensorData a ~ MaterialLeaves (TensorProduct Tensor a)
    , RunResult a ~ WithTensor TensorData a
    , BackendProgram a ~ A.Acc (BackendLeaves (TensorProduct Tensor a))
    , A.Arrays (AsyncOutput a)
    , AsyncInputs a ~ ()
    , AsyncOutput a ~ BackendLeaves (TensorProduct Tensor a)
    , RunAsyncResult a ~ IO (Async (WithTensor TensorData a))
    )
    => Runnable a
    where
    toBackendProgram = pack . flatten @a @Tensor
    fromBackendResult _ = rebuild @a @TensorData . fromBackend @(TensorProduct Tensor a)
    toBackendAsyncProgram f _ = toBackendProgram f
    fromBackendAsyncProgram shape f = Async (fromBackendResult shape) <$> f ()

{- | A runnable tensor, tuple, or record of tensors with no function arguments.

Use this constraint when the result structure is abstract.
It fixes @RunResult a@ to @WithTensor TensorData a@ without needing to inspect the concrete type @a@.
-}
type RunnableTensor a =
    ( TensorStructure a
    , BackendTree (TensorProduct Tensor a)
    , TensorProduct TensorData a ~ MaterialLeaves (TensorProduct Tensor a)
    , Runnable a
    , RunResult a ~ WithTensor TensorData a
    , RunAsyncResult a ~ IO (Async (WithTensor TensorData a))
    )

{- | Execute a closed expression or prepare a staged function for repeated execution.

For a function, bind the result outside the loop.
Reuse it with new data of the same types and shapes.
Compilation may take place on the first call.

@
result = run defaultCPU expression
step = run defaultCPU trainingStep
next = step parameters batch
@
-}
run
    :: forall f
     . Runnable f
    => Backend
    -> f
    -> RunResult f
run backend f = fromBackendResult (Proxy @f) (execute backend (toBackendProgram f))

-- | A running backend computation.
data Async a where
    Async :: (result -> a) -> Native.Async result -> Async a

-- | Wait for the computation and rethrow any exception it raised.
waitAsync
    :: Async a
    -> IO a
waitAsync (Async convert job) = convert <$> Native.wait job

-- | Return the result if ready. Rethrow any exception from a completed computation.
pollAsync
    :: Async a
    -> IO (Maybe a)
pollAsync (Async convert job) = fmap convert <$> Native.poll job

-- | Request cancellation of the task. Submitted device work may still finish.
cancelAsync
    :: Async a
    -> IO ()
cancelAsync (Async _ job) = Native.cancel job

{- | Prepare a graph for asynchronous execution. Apply all arguments to start a job.

Reuse the prepared function across calls. Its first use may compile the graph.
-}
runAsync
    :: forall f
     . Runnable f
    => Backend
    -> f
    -> RunAsyncResult f
runAsync backend f = fromBackendAsyncProgram (Proxy @f) (launch (toBackendAsyncProgram f))
  where
    launch = case backend of
        CPU DefaultCPU -> Native.run1Async
        CPU (CPUWorkers _ target) -> Native.run1AsyncWith target
#ifdef SYNAPSE_GPU
        GPU DefaultGPU -> PTX.run1Async
        GPU (GPUDevice _ target) -> PTX.run1AsyncWith target
#else
        GPU _ -> error "runAsync: Synapse was built without GPU support"
#endif

-- | Embed backend code and specialize its adapter to the original Synapse program.
compile
    :: forall f program
     . (Reflection.Typeable f, Native.Afunction program)
    => Bool
    -> TH.Name
    -> Backend
    -> program
    -> TH.ExpQ
compile asynchronous adapter backend program =
    TH.appE (pure $ TH.AppE (TH.VarE adapter) witness) (compiled backend)
  where
    witness = TH.SigE (TH.ConE 'Proxy) (TH.AppT (TH.ConT ''Proxy) programTy)
    programTy = programType $ Reflection.SomeTypeRep $ Reflection.typeRep @f

    programType (Reflection.SomeTypeRep (Reflection.Fun arg rest)) =
        TH.AppT
            (TH.AppT TH.ArrowT (programType $ Reflection.SomeTypeRep arg))
            (programType $ Reflection.SomeTypeRep rest)
    programType (Reflection.SomeTypeRep rep) = foldl TH.AppT headType (map programType args)
      where
        (con, args) = Reflection.splitApps rep
        packageName = Reflection.tyConPackage con
        moduleName = Reflection.tyConModule con
        name = Reflection.tyConName con
        headType
            | moduleName == "GHC.TypeLits"
            , Just literal <- typeLiteral name =
                TH.LitT literal
            | name == "':" = TH.PromotedConsT
            | name == "'[]" = TH.PromotedNilT
            | '\'' : constructor <- name = TH.PromotedT $ TH.mkNameG_d packageName moduleName constructor
            | otherwise = TH.ConT $ TH.mkNameG_tc packageName moduleName name
        typeLiteral value =
            (TH.NumTyLit <$> readMaybe value)
                <|> (TH.StrTyLit <$> readMaybe value)
                <|> (TH.CharTyLit <$> readMaybe value)

    compiled (CPU DefaultCPU) =
        if asynchronous then Native.runQAsync program else Native.runQ program
    compiled (CPU (CPUWorkers _ _)) =
        fail "AOT compilation requires defaultCPU for CPU programs"
#ifdef SYNAPSE_GPU
    compiled (GPU DefaultGPU) =
        if asynchronous then PTX.runQAsync program else PTX.runQ program
    compiled (GPU (GPUDevice _ _)) =
        fail "AOT compilation requires defaultGPU for GPU programs"
#else
    compiled (GPU _) =
        fail "AOT compilation: Synapse was built without GPU support"
#endif

{- | Compile a monomorphic staged program in a Template Haskell splice ahead of time.

@step = $(run' defaultCPU trainingStep)@.

Use 'defaultCPU' or 'defaultGPU' to select the backend at build time.
Configure the default target through the build environment.
Set @ACCELERATE_LLVM_NATIVE_THREADS@ for CPU workers or
@ACCELERATE_LLVM_PTX_DEVICES@ for CUDA device ordinals.

For a CPU splice, list @accelerate-llvm-native@ directly in the Cabal @build-depends@ of the component that contains the splice.
GHC loads the compiler plugin from that package when building the component.
GPU does not need this to be compiled ahead of time.
Build with @-blas@ for programs containing matrix products that would use BLAS.
Accelerate cannot embed foreign GEMM callbacks in a splice.
-}
run'
    :: forall f
     . (Runnable f, Reflection.Typeable f)
    => Backend
    -> f
    -> TH.ExpQ
run' backend = compile @f False 'fromBackendResult backend . toBackendProgram

{- | Compile a monomorphic graph for asynchronous execution in a Template Haskell splice ahead of time.

The build and runtime requirements are the same as for @run'@.
-}
runAsync'
    :: forall f
     . (Runnable f, Reflection.Typeable f)
    => Backend
    -> f
    -> TH.ExpQ
runAsync' backend = compile @f True 'fromBackendAsyncProgram backend . toBackendAsyncProgram
