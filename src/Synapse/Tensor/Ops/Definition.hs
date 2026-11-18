{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilyDependencies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Tensor primitives with custom vector-Jacobian products.
module Synapse.Tensor.Ops.Definition
    ( -- * Tensor primitives
      TensorOps (..)
    , withVJP
    , withCachedVJP

      -- * Input and output structures
    , CompatibleInput
    , Compatible

      -- * Primitive instance support
    , Wrapped
    , InputTree (..)
    , InputType (..)
    , zeroInput
    , addInput
    , Result (..)
    , ResultType (..)
    , resultValues
    , resultFields
    ) where

import qualified Data.Array.Accelerate as A
import Data.Bifunctor (bimap)
import Data.Kind (Constraint, Type)
import Data.Proxy (Proxy)
import Synapse.Tensor.Ops.Elementwise ()
import qualified Synapse.Tensor.Ops.Raw.Construction as RawConstruction
import qualified Synapse.Tensor.Ops.Raw.Elementwise as RawElementwise
import Synapse.Tensor.Representation (DType)
import qualified Synapse.Tensor.Representation as T
import Synapse.Tensor.Shapes.Dimensions (Shape)
import Synapse.Tensor.Structure
    ( TensorField (..)
    , TensorProduct
    , TensorStructure
    )
import qualified Synapse.Tensor.Structure as T
import Synapse.Tensor.Structure.Product (ProductFields, ProductValues)

type Wrapped t a = ProductFields t a

-- All registered input leaves share the TensorOps scalar type.
data InputTree a input where
    InputTensor
        :: Shape sh => InputTree a (TensorField (T.Tensor sh a))
    InputPair
        :: InputTree a l
        -> InputTree a r
        -> InputTree a (l, r)

deriving stock instance Show (InputTree a input)

class Wrapped T.Tensor input ~ input => InputType a input | input -> a where
    inputTree
        :: InputTree a input

instance Shape sh => InputType a (TensorField (T.Tensor sh a)) where inputTree = InputTensor
instance (InputType a l, InputType a r) => InputType a (l, r) where
    inputTree = InputPair (inputTree @a @l) (inputTree @a @r)

zeroInput
    :: (T.DType a, Num a, Num (T.TensorElement a))
    => InputTree a input
    -> ProductValues input
zeroInput InputTensor = RawConstruction.full 0
zeroInput (InputPair l r) = (zeroInput l, zeroInput r)

addInput
    :: (T.DType a, Num a, Num (T.TensorElement a))
    => InputTree a input
    -> ProductValues input
    -> ProductValues input
    -> ProductValues input
addInput InputTensor = RawElementwise.zipWith (+)
addInput (InputPair l r) = \(a, b) (c, d) -> (addInput l a c, addInput r b d)

-- | Single results may change element type without requiring numeric output evidence.
data Result a output where
    Single
        :: Result a (TensorField (T.Tensor sh b))
    Product
        :: InputTree a output
        -> Result a output

deriving stock instance Show (Result a output)

class Wrapped T.Tensor output ~ output => ResultType a output where
    resultTree
        :: Result a output

instance ResultType a (TensorField (T.Tensor sh b)) where resultTree = Single
instance (InputType a l, InputType a r) => ResultType a (l, r) where
    resultTree = Product $ InputPair (inputTree @a @l) (inputTree @a @r)

-- | Extract tensor values before registering a primitive result.
resultValues
    :: Result a output
    -> output
    -> ProductValues output
resultValues Single = tensorFieldValue
resultValues (Product tree) = inputValues tree
  where
    inputValues
        :: InputTree a input
        -> input
        -> ProductValues input
    inputValues InputTensor = tensorFieldValue
    inputValues (InputPair left right) = bimap (inputValues left) (inputValues right)

-- | Wrap primitive result values for reconstruction.
resultFields
    :: Result a output
    -> ProductValues output
    -> output
resultFields Single = TensorField []
resultFields (Product tree) = inputFields tree
  where
    inputFields
        :: InputTree a input
        -> ProductValues input
        -> input
    inputFields InputTensor = TensorField []
    inputFields (InputPair left right) = bimap (inputFields left) (inputFields right)

{- | Tensor input structures accepted by primitive computations.
Resolved automatically for tensors, tuples, and generic tensor records.
-}
type CompatibleInput (t :: T.TensorLike) a input =
    (TensorStructure input, InputType a (TensorProduct T.Tensor input))

{- | Tensor-only input and output structures accepted by 'customVJP'.
Resolved automatically for individual tensors, tuples, and generic records
with a 'T.WithTensor' correspondence. Inputs share the operation's element
type. Multiple outputs use that same element type; a single tensor result
may use another element type.
-}
type Compatible t a input output =
    (CompatibleInput t a input, TensorStructure output, ResultType a (TensorProduct T.Tensor output))

-- | Tensor representations supporting custom vector-Jacobian products.
class
    (DType a, BackwardConstraints t (), BackwardConstraints t (Num (T.TensorElement a))) =>
    TensorOps (t :: T.TensorLike) a
    where
    -- | Element-expression constructor corresponding to the tensor representation.
    type Element t = (e :: Type -> Type) | e -> t

    {- | Requirements of an individual reverse rule for this tensor representation.
    Ordinary tensors ignore the constraint; differentiable tensors require it.
    Every representation supports the empty request and a request for numeric element cotangents,
    which differentiation uses to accumulate contributions from separate paths.
    The constraint is selected separately for each call to 'customVJP' or 'customElementVJP'.
    -}
    type BackwardConstraints t (c :: Constraint) :: Constraint

    -- | Embed a constant tensor.
    fromTensor
        :: T.Tensor sh a
        -> t sh a

    -- | Extract the underlying tensor value.
    toTensor
        :: t sh a
        -> T.Tensor sh a

    -- | Convert an element expression to a scalar tensor.
    fromElement
        :: Element t a
        -> t '[] a

    -- | Apply an element-valued primitive with a custom vector-Jacobian product.
    customElementVJP
        :: (CompatibleInput t a input, BackwardConstraints t c)
        => Proxy c
        -> (input -> T.TensorElement b)
        -> (input -> (T.TensorElement b, cache))
        -> (c => cache -> T.TensorElement b -> input)
        -> T.WithTensor t input
        -> Element t b

    -- | Apply a primitive with a custom vector-Jacobian product, retaining the cache needed by that product.
    customVJP
        :: forall input output cache c
         . (Compatible t a input output, BackwardConstraints t c)
        => Proxy c
        -> (input -> output)
        -> (input -> (output, cache))
        -> (c => cache -> output -> input)
        -> T.WithTensor t input
        -> T.WithTensor t output

-- | Define a reverse rule using the original input, forward output, and output gradient.
withVJP
    :: forall t a input output c
     . (TensorOps t a, Compatible t a input output, BackwardConstraints t c)
    => Proxy c
    -> (input -> output)
    -> (c => input -> output -> output -> input)
    -> T.WithTensor t input
    -> T.WithTensor t output
withVJP constraints forward backward =
    customVJP @t @a
        constraints
        forward
        (\input -> let output = forward input in (output, (input, output)))
        (\(input, output) seed -> backward input output seed)

-- | Define a reverse rule using an explicit cache and the output gradient.
withCachedVJP
    :: forall t a input output cache c
     . (TensorOps t a, Compatible t a input output, BackwardConstraints t c)
    => Proxy c
    -> (input -> (output, cache))
    -> (c => cache -> output -> input)
    -> T.WithTensor t input
    -> T.WithTensor t output
withCachedVJP constraints forward = customVJP @t @a constraints (fst . forward) forward

instance T.DType a => TensorOps T.Tensor a where
    type Element T.Tensor = T.TensorElement
    type BackwardConstraints T.Tensor c = ()
    fromTensor = id
    toTensor = id
    fromElement (T.TensorElement x) = T.Tensor (A.unit x)
    customElementVJP _ forward _ _ = forward
    customVJP _ forward _ _ = forward
