module SGDInt where

import qualified Synapse.NN as NN

bad :: (Int, ())
bad = NN.optimizerStep (NN.SGD 0.1) (1 :: Int) 1 ()
