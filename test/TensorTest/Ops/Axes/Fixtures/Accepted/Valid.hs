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

scalarPermutation :: T.Tensor '[] Int
scalarPermutation = T.permuteAxes @'[] (T.scalar 7)

appendDimension :: T.Tensor '[2, 3, 1] Int
appendDimension = T.unsqueeze @2 (T.zeros @'[2, 3])

scalarExpansion :: T.Tensor '[1] Int
scalarExpansion = T.unsqueeze @0 (T.scalar 7)

scalarSqueeze :: T.Tensor '[] Int
scalarSqueeze = T.squeeze @0 (T.ones @'[1])

reordered :: T.Tensor '[4, 2, 3] Int
reordered = T.permuteAxes @'[2, 0, 1] (T.zeros @'[2, 3, 4])

unsqueezed :: T.Tensor '[2, 1, 3] Int
unsqueezed = T.unsqueeze @1 (T.zeros @'[2, 3])
