-- | Supported tensor element types.
--
-- The 'DType' class is Synapse's public constraint for values that can be stored
-- inside tensors.
module Synapse.Tensor.DType
  ( DType,
    Float32,
    Float64,
  )
where

import qualified Data.Array.Accelerate as A
import Data.Int (Int32, Int64)
import Data.Word (Word32, Word64)

-- | Single-precision floating point tensor element.
type Float32 = Float

-- | Double-precision floating point tensor element.
type Float64 = Double

-- | Element types supported by Synapse tensors.
class (A.Elt a) => DType a

instance DType Float

instance DType Double

instance DType Int

instance DType Int32

instance DType Int64

instance DType Word32

instance DType Word64

instance DType Bool
