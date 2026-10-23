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
