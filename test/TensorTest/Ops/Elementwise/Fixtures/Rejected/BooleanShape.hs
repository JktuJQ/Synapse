{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module BooleanShape where

import qualified Synapse.Tensor as T

bad = T.full @'[2] True T.&& T.full @'[3] False
