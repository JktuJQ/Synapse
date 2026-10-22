{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Valid where

import qualified Synapse.Tensor as T

element :: T.TensorElement Bool
element = T.constant (1 :: Int) T.< T.constant 2

tensor :: T.Tensor '[2, 3] Bool
tensor = (T.ones @'[2, 3] @Int :: T.Tensor '[2, 3] Int) T.== T.zeros @'[2, 3]

extrema :: T.Tensor '[2, 3] Float
extrema = T.max (T.zeros @'[2, 3]) (T.ones @'[2, 3])
