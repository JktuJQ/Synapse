{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmptyMaximumAxis where

import qualified Synapse.Tensor as T

bad = T.maximumAxes @'[0] (T.zeros @'[0, 3] @Float)
