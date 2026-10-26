{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SGDInt where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

bad :: (T.Tensor '[] Int, ())
bad = NN.optimizerStep (NN.SGD 0.1) (T.ones @'[] @Int) 1 ()
