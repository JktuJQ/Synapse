{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Valid where

import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes

selection :: (Axes.CanSelectAxes '[2, 0] '[2, 3, 4]) => ()
selection = ()

validSelection :: ()
validSelection = selection

appendDimension :: T.Tensor '[2, 3, 1] Int
appendDimension = T.unsqueeze @2 (T.zeros @'[2, 3])

scalarExpansion :: T.Tensor '[1] Int
scalarExpansion = T.unsqueeze @0 (T.scalar 7)

scalarSqueeze :: T.Tensor '[] Int
scalarSqueeze = T.squeeze @0 (T.ones @'[1])

unsqueezed :: T.Tensor '[2, 1, 3] Int
unsqueezed = T.unsqueeze @1 (T.zeros @'[2, 3])

generated :: T.Tensor '[2, 3] Int
generated = T.fromGen @'[2, 3] (\ix -> T.coordinate @0 ix * 10 + T.coordinate @1 ix)

permuted :: T.Tensor '[4, 2, 3] Int
permuted = T.permuteAxes @'[2, 0, 1] (T.zeros @'[2, 3, 4])

transposed :: T.Tensor '[2, 8, 4, 3] Float
transposed = T.transpose (T.zeros @'[2, 8, 3, 4])

permutedScalar :: T.Tensor '[] Bool
permutedScalar = T.permuteAxes @'[] (T.scalar (T.constant True))

joined :: T.Tensor '[2, 7] Int
joined = T.concat @1 (T.zeros @'[2, 3]) (T.zeros @'[2, 4])

stacked :: T.Tensor '[2, 3, 2] Float
stacked = T.stack @2 (T.zeros @'[2, 3]) (T.zeros @'[2, 3])

stackedScalars :: T.Tensor '[2] Bool
stackedScalars = T.stack @0 (T.scalar (T.constant True)) (T.scalar (T.constant False))
