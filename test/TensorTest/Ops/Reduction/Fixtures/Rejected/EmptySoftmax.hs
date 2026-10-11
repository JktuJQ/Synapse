{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmptySoftmax where

import qualified Synapse.Tensor as T

bad = T.softmax @1 (T.zeros @'[2, 0] @Float)
