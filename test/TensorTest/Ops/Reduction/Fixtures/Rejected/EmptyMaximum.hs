{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmptyMaximum where

import qualified Synapse.Tensor as T

bad = T.maximum (T.zeros @'[0] @Double)
