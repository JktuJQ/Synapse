{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module SqueezeZero where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (natVal)
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes

bad = T.squeeze @0 (T.zeros @'[0, 3] :: T.Tensor '[0, 3] Int)
