{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module IntegerReshape where

import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T

bad :: (AG.Context s) => AG.Tensor s '[2] Int -> AG.Tensor s '[1, 2] Int
bad = T.reshape @'[1, 2]
