{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module SelectBounds where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (natVal)
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes

bad = T.shape @(Axes.SelectAxes '[0, 3] '[2, 3, 4])
