{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module EmptyMinimumAxis where

import qualified Synapse.Tensor as T

bad = T.minimumAxes @'[1] (T.zeros @'[2, 0, 3] @Int)
