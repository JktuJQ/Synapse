{-# LANGUAGE CPP #-}
{-# LANGUAGE TypeFamilies #-}

-- | BLAS calls for contiguous row-major matrix products.
module Synapse.Tensor.Ops.Raw.GEMM (matrixProduct) where

import Data.Array.Accelerate qualified as A
import Synapse.Tensor.Representation (DType)

#ifdef SYNAPSE_BLAS
import Control.Monad (forM_)
import Data.Array.Accelerate.Array.Data (ArrayData)
import Data.Array.Accelerate.Array.Unique (UniqueArray, withUniqueArrayPtr)
import Data.Array.Accelerate.LLVM.Native.Foreign qualified as Native
import Data.Array.Accelerate.Representation.Array (Array (..), ArrayR (..))
import Data.Array.Accelerate.Representation.Shape (DIM1, dim1)
import Data.Array.Accelerate.Sugar.Elt (EltR, eltR)
import Data.Type.Equality ((:~:) (Refl))
import Data.Typeable (eqT)
import Foreign.C.Types (CInt (..))
import Foreign.Marshal.Array (advancePtr)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable)

#ifdef SYNAPSE_GPU
import Control.Exception (bracket)
import Data.Array.Accelerate.Lifetime (withLifetime)
import Data.Array.Accelerate.LLVM.PTX.Foreign qualified as PTX
import Data.Array.Accelerate.Array.Data (ScalarArrayDataR)
import Data.Array.Accelerate.Type (SingleType, singleType)
import Foreign.CUDA.Driver.Stream qualified as CUDA
import Foreign.CUDA.Ptr qualified as CUDA
import Foreign.Marshal.Alloc (alloca)
import Foreign.Marshal.Utils (with)
import Foreign.Storable (peek)
#endif
#endif

#ifdef SYNAPSE_BLAS
-- | Matrix dimensions, batch strides, and operand orientations.
data Layout = Layout
    { batchCount :: Int
    , rowCount :: Int
    , innerCount :: Int
    , columnCount :: Int
    , strideLeft :: Int
    , strideRight :: Int
    , transposedLeft :: Bool
    , transposedRight :: Bool
    }
    deriving stock (Show)

-- | Pair of flattened arrays in backend representation.
type Operands a = (((), Array DIM1 a), Array DIM1 a)

-- | Leading dimension of the stored left matrix.
leadingLeft
    :: Layout
    -> CInt
leadingLeft layout = fromIntegral $ if transposedLeft layout then rowCount layout else innerCount layout

-- | Leading dimension of the stored right matrix.
leadingRight
    :: Layout
    -> CInt
leadingRight layout = fromIntegral $ if transposedRight layout then innerCount layout else columnCount layout

-- | Row-major CBLAS GEMM signature.
type CBLAS a =
    CInt -> CInt -> CInt -> CInt -> CInt -> CInt -> a ->
    Ptr a -> CInt -> Ptr a -> CInt -> a -> Ptr a -> CInt -> IO ()

foreign import ccall safe "cblas_sgemm" cblasSgemm :: CBLAS Float
foreign import ccall safe "cblas_dgemm" cblasDgemm :: CBLAS Double

-- | Multiply each pair of matrices in the batch.
runCBLAS
    :: (Num a, Storable a)
    => Layout
    -> CBLAS a
    -> Ptr a
    -> Ptr a
    -> Ptr a
    -> IO ()
runCBLAS layout gemm x y z =
    forM_ [0 .. batchCount layout - 1] $ \batch ->
        gemm 101 (cpuOp $ transposedLeft layout) (cpuOp $ transposedRight layout)
            (fromIntegral $ rowCount layout) (fromIntegral $ columnCount layout)
            (fromIntegral $ innerCount layout) 1
            (advancePtr x (batch * strideLeft layout)) (leadingLeft layout)
            (advancePtr y (batch * strideRight layout)) (leadingRight layout)
            0 (advancePtr z (batch * outputStride)) (fromIntegral $ columnCount layout)
  where
    cpuOp transposed = if transposed then 112 else 111
    outputStride = rowCount layout * columnCount layout

-- | Allocate the output and run the batch on the CPU.
cpuProduct
    :: forall a
     . (DType a, Num a, Storable a, EltR a ~ a, ArrayData a ~ UniqueArray a)
    => Layout
    -> CBLAS a
    -> Native.ForeignAcc (Operands a -> Array DIM1 a)
cpuProduct layout gemm = Native.ForeignAcc ("synapse.cblas.gemm" ++ show layout) $ \(((), Array _ xs), Array _ ys) -> do
    result@(Array _ zs) <- Native.allocateRemote (ArrayR dim1 (eltR @a)) ((), batchCount layout * outputStride)
    Native.liftIO $
        withUniqueArrayPtr xs $ \x ->
            withUniqueArrayPtr ys $ \y ->
                withUniqueArrayPtr zs $ \z ->
                    runCBLAS layout gemm x y z
    future <- Native.new
    Native.put future result
    pure future
  where
    outputStride = rowCount layout * columnCount layout

#ifdef SYNAPSE_GPU
-- | Column-major cuBLAS GEMM signature.
type CuBLAS a =
    Ptr () -> CInt -> CInt -> CInt -> CInt -> CInt ->
    Ptr a -> Ptr a -> CInt -> Ptr a -> CInt -> Ptr a -> Ptr a -> CInt -> IO CInt

foreign import ccall unsafe "cublasCreate_v2" cublasCreate :: Ptr (Ptr ()) -> IO CInt
foreign import ccall unsafe "cublasDestroy_v2" cublasDestroy :: Ptr () -> IO CInt
foreign import ccall unsafe "cublasSetStream_v2" cublasSetStream :: Ptr () -> Ptr () -> IO CInt
foreign import ccall unsafe "cublasSgemm_v2" cublasSgemm :: CuBLAS Float
foreign import ccall unsafe "cublasDgemm_v2" cublasDgemm :: CuBLAS Double

-- | Raise an IO error for a failing cuBLAS call.
check
    :: String
    -> IO CInt
    -> IO ()
check operation action = do
    status <- action
    if status == 0
        then pure ()
        else ioError $ userError $ "Synapse cuBLAS " ++ operation ++ " failed with status " ++ show status

-- | Create a cuBLAS handle in the current CUDA context.
createHandle
    :: IO (Ptr ())
createHandle = alloca $ \out -> check "create" (cublasCreate out) >> peek out

-- | Multiply each row-major pair by reversing the operand order for cuBLAS.
runCuBLAS
    :: (Num a, Storable a)
    => Layout
    -> CuBLAS a
    -> Ptr ()
    -> Ptr a
    -> Ptr a
    -> Ptr a
    -> IO ()
runCuBLAS layout gemm stream x y z =
    bracket createHandle (check "destroy" . cublasDestroy) $ \handle -> do
        check "setStream" (cublasSetStream handle stream)
        with 1 $ \alpha ->
            with 0 $ \beta ->
                forM_ [0 .. batchCount layout - 1] $ \batch ->
                    check "gemm" $
                        gemm handle (gpuOp $ transposedRight layout) (gpuOp $ transposedLeft layout)
                            (fromIntegral $ columnCount layout) (fromIntegral $ rowCount layout) (fromIntegral $ innerCount layout)
                            alpha (advancePtr y (batch * strideRight layout)) (leadingRight layout)
                            (advancePtr x (batch * strideLeft layout)) (leadingLeft layout)
                            beta (advancePtr z (batch * outputStride)) (fromIntegral $ columnCount layout)
  where
    gpuOp transposed = if transposed then 1 else 0
    outputStride = rowCount layout * columnCount layout

-- | Allocate the output and enqueue the batch on the current CUDA stream.
gpuProduct
    :: forall a
     . (DType a, Num a, Storable a, EltR a ~ a, ScalarArrayDataR a ~ a)
    => Layout
    -> SingleType a
    -> CuBLAS a
    -> PTX.ForeignAcc (Operands a -> Array DIM1 a)
gpuProduct layout scalar gemm = PTX.ForeignAcc ("synapse.cublas.gemm" ++ show layout) $ \(((), Array _ xs), Array _ ys) -> do
    stream <- PTX.asksParState PTX.ptxStream
    result@(Array _ zs) <- PTX.allocateRemote (ArrayR dim1 (eltR @a)) ((), batchCount layout * rowCount layout * columnCount layout)
    let withBuffer buffer action = PTX.withDevicePtr scalar buffer $ \ptr -> do
            value <- action (CUDA.useDevicePtr ptr)
            event <- PTX.waypoint stream
            pure (Just event, value)
    PTX.liftPar $
        withBuffer xs $ \x ->
            withBuffer ys $ \y ->
                withBuffer zs $ \z ->
                    PTX.liftIO $ withLifetime stream $ \cudaStream ->
                        runCuBLAS layout gemm (CUDA.useStream cudaStream) x y z
    future <- PTX.new
    PTX.put future result
    pure future

#endif
#endif

-- | Dispatch supported contiguous products to BLAS.
matrixProduct
    :: forall a
     . DType a
    => Int
    -> Int
    -> Int
    -> Int
    -> Int
    -> Int
    -> Bool
    -> Bool
    -> A.Acc (A.Vector a)
    -> A.Acc (A.Vector a)
    -> (A.Acc (A.Vector a) -> A.Acc (A.Vector a) -> A.Acc (A.Vector a))
    -> A.Acc (A.Vector a)
#ifdef SYNAPSE_BLAS
matrixProduct batches rows shared cols leftStride rightStride transposeLeft transposeRight left right fallback
    | not (all fits [rows, shared, cols]) || batches <= 0 = fallback left right
    | Just Refl <- eqT @a @Float =
        A.foreignAcc (cpuProduct layout cblasSgemm)
#ifdef SYNAPSE_GPU
            (A.foreignAcc (gpuProduct layout (singleType @Float) cublasSgemm) fallbackPair)
#else
            fallbackPair
#endif
            operands
    | Just Refl <- eqT @a @Double =
        A.foreignAcc (cpuProduct layout cblasDgemm)
#ifdef SYNAPSE_GPU
            (A.foreignAcc (gpuProduct layout (singleType @Double) cublasDgemm) fallbackPair)
#else
            fallbackPair
#endif
            operands
    | otherwise = fallback left right
  where
    fits n = n > 0 && toInteger n <= toInteger (maxBound :: CInt)
    layout = Layout batches rows shared cols leftStride rightStride transposeLeft transposeRight
    operands :: A.Acc (A.Vector a, A.Vector a)
    operands = A.T2 left right
    fallbackPair :: A.Acc (A.Vector a, A.Vector a) -> A.Acc (A.Vector a)
    fallbackPair (A.T2 x y) = fallback x y
#else
matrixProduct _ _ _ _ _ _ _ _ left right fallback = fallback left right
#endif
