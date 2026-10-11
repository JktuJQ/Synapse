{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmptyArgmax where

import qualified Synapse.Tensor as T

bad = T.argmax @1 (T.zeros @'[2, 0] @Float)
