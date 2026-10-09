{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

module SelectConstraintDuplicate where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (natVal)
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes

proof :: (Axes.CanSelectAxes '[1, 1] '[2, 3]) => ()
proof = ()

bad :: ()
bad = proof
