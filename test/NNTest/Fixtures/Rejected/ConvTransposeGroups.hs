{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ConvTransposeGroups where

import qualified Synapse.NN.Layers as NN
import qualified Synapse.Tensor as T

bad = NN.convTranspose1d @'[1] @'[0, 0] @'[1] @'[0] (NN.TransposedConvolution (T.zeros @'[2, 0, 1]) (T.zeros @'[1]) :: NN.TransposedConvolution 2 1 '[1] 2 T.Tensor Double) (T.ones @'[1, 2, 2])
