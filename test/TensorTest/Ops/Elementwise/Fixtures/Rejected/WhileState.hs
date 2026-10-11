module WhileState where

import qualified Synapse.Tensor as T

bad = T.while (\_ -> T.constant True) (\_ -> T.constant False) (T.constant (0 :: Int))
