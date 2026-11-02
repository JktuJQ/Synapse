{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module ConvKernel where

import qualified Synapse.NN.Layers as NN
import qualified Synapse.Tensor as T

bad = NN.conv2d @'[1, 1] @'[0, 0, 0, 0] @'[1, 1] (NN.Convolution (T.ones @'[1, 1, 1]) (T.zeros @'[1]) :: NN.Convolution 1 1 '[1] 1 T.Tensor Double) (T.ones @'[1, 1, 2, 2])
