{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

module SelectConstraintBounds where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (natVal)
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes

proof :: (Axes.CanSelectAxes '[3] '[2, 3, 4]) => ()
proof = ()

bad :: ()
bad = proof
