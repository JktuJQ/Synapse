{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Valid where

import qualified Synapse.Tensor as T

element :: T.TensorElement Int
element = T.index @'[1, 2] (T.zeros @'[2, 3])

materialized :: T.Tensor '[] Int
materialized = T.scalar (element + 1)

row :: T.Tensor '[3] Int
row = T.slice @'[ 'T.At 1, 'T.All] (T.zeros @'[2, 3])

strided :: T.Tensor '[2, 3] Int
strided = T.slice @'[ 'T.All, 'T.Range 1 8 3] (T.zeros @'[2, 8])

reversed :: T.Tensor '[2, 3] Int
reversed = T.slice @'[ 'T.RangeReverse 0 2 1, 'T.All] (T.zeros @'[2, 3])

reverseRange :: T.Tensor '[3] Int
reverseRange = T.slice @'[ 'T.RangeReverse 1 7 2] (T.zeros @'[8])

scalar :: T.TensorElement Bool
scalar = T.index @'[] (T.scalar (T.element True))

booleanRow :: T.Tensor '[3] Bool
booleanRow = T.slice @'[ 'T.At 1, 'T.All] (T.full @'[2, 3] True)

booleanBroadcast :: T.Tensor '[2, 3] Bool
booleanBroadcast = T.broadcast (T.full @'[3] True)

numericSum :: T.NumDType a => T.Tensor '[3] a -> T.Tensor '[] a
numericSum = T.sum
