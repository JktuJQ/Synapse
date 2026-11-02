{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ConvFeatures where

import qualified Synapse.NN.Layers as NN
import qualified Synapse.Tensor as T

bad = NN.conv1d @'[1] @'[0, 0] @'[1] (NN.Convolution (T.ones @'[2, 1, 1]) (T.zeros @'[2]) :: NN.Convolution 1 2 '[1] 1 T.Tensor Double) (T.ones @'[1, 3, 3])
