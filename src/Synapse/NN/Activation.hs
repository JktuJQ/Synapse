{- | Shape-preserving activation functions for tensors.
Activations support ordinary function application and composition in an arbitrary arrow.
-}
module Synapse.NN.Activation (relu, leakyRelu, sigmoid, tanh, sin, softplus) where

import Control.Arrow (Arrow, arr)
import Data.Kind (Constraint)
import Data.Proxy (Proxy (..))
import Synapse.Tensor.Ops.Constraints (FloatingDType, NumDType, OrdDType)
import Synapse.Tensor.Ops.Definition (TensorOps, withCachedVJP, withVJP)
import Synapse.Tensor.Ops.Logic qualified as E
import Synapse.Tensor.Ops.Math ()
import Synapse.Tensor.Representation (Tensor, TensorLike)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Prelude hiding (sin, tanh)
import Prelude qualified as P

-- | Apply the rectified linear activation @relu x = max 0 x@ elementwise.
relu
    :: forall {arrow} {t :: TensorLike} sh a
     . (Arrow arrow, TensorOps t a, Shape sh, NumDType a, OrdDType a)
    => arrow (t sh a) (t sh a)
relu = arr $ withVJP (Proxy @(() :: Constraint)) forward backward
  where
    forward x = E.select (x E.<= 0) 0 x
    backward x _ seed = E.select (x E.<= 0) 0 seed

{- | Apply @x@ when @x > 0@ and @slope * x@ otherwise, elementwise.
The first argument controls the slope for negative inputs.
For example, @leakyRelu 0.01 input@ retains one percent of each negative input.
-}
leakyRelu
    :: forall {arrow} {t :: TensorLike} sh a
     . (Arrow arrow, TensorOps t a, Shape sh, NumDType a, OrdDType a)
    => t sh a
    -> arrow (t sh a) (t sh a)
leakyRelu slope = arr $ \input ->
    withVJP (Proxy @(() :: Constraint)) forward backward (slope, input)
  where
    forward (a, x) = E.select (x E.> 0) x (a * x)
    backward (a, x) _ seed =
        let positive = x E.> 0
         in (E.select positive 0 (seed * x), E.select positive seed (seed * a))

-- | Apply the logistic sigmoid @sigmoid x = 1 / (1 + exp (-x))@ elementwise.
sigmoid
    :: forall {arrow} {t :: TensorLike} sh a
     . (Arrow arrow, TensorOps t a, Shape sh, FloatingDType a, OrdDType a)
    => arrow (t sh a) (t sh a)
sigmoid = arr $ withCachedVJP (Proxy @(() :: Constraint)) forward (*)
  where
    forward
        :: Tensor sh a
        -> (Tensor sh a, Tensor sh a)
    forward input =
        let z = exp (negate (abs input))
            large = recip (1 + z)
            small = z * large
         in (E.select (input E.>= 0) large small, large * small)

-- | Apply the hyperbolic tangent @tanh x = (exp x - exp (-x)) / (exp x + exp (-x))@ elementwise.
tanh
    :: forall {arrow} {t :: TensorLike} sh a
     . (Arrow arrow, Floating (t sh a))
    => arrow (t sh a) (t sh a)
tanh = arr P.tanh

-- | Apply the sine activation @sin x@ elementwise, with angles in radians.
sin
    :: forall {arrow} {t :: TensorLike} sh a
     . (Arrow arrow, Floating (t sh a))
    => arrow (t sh a) (t sh a)
sin = arr P.sin

-- | Apply @softplus x = log (1 + exp x)@ elementwise as a smooth approximation to 'relu'.
softplus
    :: forall {arrow} {t :: TensorLike} sh a
     . (Arrow arrow, TensorOps t a, Shape sh, FloatingDType a, OrdDType a)
    => arrow (t sh a) (t sh a)
softplus = arr $ withCachedVJP (Proxy @(() :: Constraint)) forward (*)
  where
    forward
        :: Tensor sh a
        -> (Tensor sh a, Tensor sh a)
    forward input =
        let positive = input E.> 0
            z = exp (negate (abs input))
            correction = 2 * atanh (z / (2 + z))
            large = recip (1 + z)
         in (E.select positive input 0 + correction, E.select positive large (z * large))
