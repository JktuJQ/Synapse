{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module OperatorAxis where

import Data.Proxy (Proxy (..))
import qualified Synapse.Tensor as T

bad = T.fromGen @'[2, 3] (\ix -> ix T.!! Proxy @2)
