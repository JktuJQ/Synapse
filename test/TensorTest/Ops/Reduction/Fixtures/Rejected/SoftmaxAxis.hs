{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module SoftmaxAxis where

import qualified Synapse.Tensor as T

bad = T.softmax @2 (T.zeros @'[2, 3] @Double)
