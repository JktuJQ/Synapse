{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE KindSignatures #-}

-- | Models with statically determined input and output shapes.
module Synapse.NN.Model (Model (..)) where

import Data.Kind (Type)
import GHC.TypeLits (Nat)
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Type (TensorLike)

-- | A model that accepts a particular input shape and tensor representation.
-- Operation constraints belong to instances, allowing the same 'forward' to serve inference and differentiation.
-- The model and input shape determine the output shape independently of the tensor representation and element type.
class
  (Shape in_sh, Shape out_sh) =>
  Model (model :: TensorLike -> Type -> Type) (in_sh :: [Nat]) (out_sh :: [Nat]) (t :: TensorLike) a
    | model in_sh -> out_sh
  where
  -- | Apply the model's parameters to an input tensor.
  forward :: model t a -> t in_sh a -> t out_sh a
