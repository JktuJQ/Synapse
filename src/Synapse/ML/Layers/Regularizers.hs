{- | Allows to regularize values of layers parameters.

@Regularizer@ type alias represents functions that are able to add penalties on loss depending on the state of parameters matrices.
-}


module Synapse.ML.Layers.Regularizers
    ( -- * @Regularizer@ type alias

      Regularizer
    
      -- * Regularizers functions
    
    , l1
    , l2
    ) where


import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M

import Data.Foldable (toList)


-- | @Regularizer@ type alias represents functions that are able to add penalties on loss depending on the state of parameters matrices.
type Regularizer a = Mat a -> a


-- Regularizers

-- | @l1@ applies a L1 regularization penalty (sum of absolute values of matrix multiplied by coefficient).
l1 :: Num a => a -> Regularizer a
l1 k = (* k) . sum . toList . abs

-- | @l2@ applies a L2 regularization penalty (sum of squared values of matrix multiplied by coefficient).
l2 :: Num a => a -> Regularizer a
l2 k mat = (* k) . sum $ toList $ M.elementwise (*) mat mat
