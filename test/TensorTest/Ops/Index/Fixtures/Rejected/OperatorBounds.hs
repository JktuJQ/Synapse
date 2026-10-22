{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module OperatorBounds where

import Data.Proxy (Proxy (..))
import qualified Synapse.Tensor as T

bad = (T.ones @'[2, 3] @Int :: T.Tensor '[2, 3] Int) T.!! Proxy @'[0, 3]
