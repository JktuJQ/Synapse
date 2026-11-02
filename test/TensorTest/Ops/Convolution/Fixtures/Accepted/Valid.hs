{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Valid where

import qualified Synapse.Tensor as T

valid :: T.Tensor '[2, 3, 2, 3] Int
valid = T.conv2d @'[2, 1] @'[1, 0, 0, 1] @'[1, 1] @1 (T.ones @'[2, 2, 3, 4]) (T.ones @'[3, 2, 2, 3])
