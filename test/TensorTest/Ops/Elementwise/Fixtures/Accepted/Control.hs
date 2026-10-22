{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Control where

import qualified Synapse.Tensor as T

mask :: T.Tensor '[2] Bool
mask = T.not (T.full @'[2] True) T.|| T.full @'[2] False

selected :: T.Tensor '[2] Int
selected = T.select mask (T.ones @'[2]) (T.zeros @'[2])

branched :: T.Tensor '[2] Bool
branched = T.cond (T.scalar (T.constant True)) mask (T.not mask)

loop :: T.TensorElement Int
loop = T.while (T.< 5) (+ 1) 0

iterated :: T.TensorElement Int
iterated = T.iterate 3 (* 2) 1

tuple :: T.TensorElement (Int, Bool)
tuple = T.select (T.constant True) (T.pair 3 (T.constant True)) (T.pair 7 (T.constant False))

projection :: T.TensorElement Int
projection = T.fst tuple
