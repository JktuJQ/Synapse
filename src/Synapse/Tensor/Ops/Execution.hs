-- | Control evaluation of tensor operations.
module Synapse.Tensor.Ops.Execution (force) where

import Data.Array.Accelerate qualified as A
import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Ops.Definition (TensorOps, withVJP)
import Synapse.Tensor.Representation (Tensor (..))
import Synapse.Tensor.Shapes.Dimensions (Shape)

-- | Prevent fusion across this point by evaluating the tensor before its consumers.
force
    :: (TensorOps t a, Shape sh)
    => t sh a
    -> t sh a
force = withVJP (Proxy @(() :: Constraint)) (\(Tensor input) -> Tensor (A.compute input)) (\_ _ seed -> seed)
