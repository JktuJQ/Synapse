{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module PermuteBounds where

import Data.Proxy (Proxy (..))
import GHC.TypeLits (natVal)
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Shapes.Axes as Axes

bad = T.permuteAxes @'[0, 2] (T.zeros @'[2, 3] :: T.Tensor '[2, 3] Int)
