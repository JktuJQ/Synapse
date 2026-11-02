{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module LayerNormSuffix where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[2, 3] Double
invalid = NN.layerNorm 1 (NN.LayerNorm (T.full @'[1] 1) 0) (T.zeros @'[2, 3])
