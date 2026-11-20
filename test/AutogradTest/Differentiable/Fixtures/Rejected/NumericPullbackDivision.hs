module NumericPullbackDivision where

import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import qualified Synapse.Tensor as T
import Synapse.Tensor.Ops (TensorOps (customVJP))

bad :: (TensorOps t a, T.Shape sh, Num (T.TensorElement a)) => t sh a -> t '[] a
bad =
    customVJP (Proxy @(() :: Constraint)) T.sum (\input -> (T.sum input, ())) (\() seed -> T.broadcast $ seed / 2)
