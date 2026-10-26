{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}

module IntegerArithmetic where

import qualified Synapse.Autograd as AG

bad :: (AG.Context s) => AG.Tensor s '[2] Int -> AG.Tensor s '[2] Int
bad x = x + x
