{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module BatchNormChannels where

import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T

invalid :: T.Tensor '[2, 2] Double
invalid = NN.evalState (NN.batchNorm @1 NN.Eval 1 0.1 (NN.BatchNorm (T.full @'[3] 1) 0) $ T.zeros @'[2, 2]) NN.initialBatchNormState
