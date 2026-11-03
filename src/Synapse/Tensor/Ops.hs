{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Primitive tensor computations with custom reverse-mode derivatives.
module Synapse.Tensor.Ops (TensorOps (..), Needs (..), Compatible, CompatibleInput) where

import qualified Data.Array.Accelerate as A
import Data.Kind (Constraint)
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Ops.Structure
import Synapse.Tensor.Type (DType)
import qualified Synapse.Tensor.Type as T

-- | Capabilities required by a primitive's reverse rule.
-- Requests do not impose these constraints on ordinary tensor evaluation.
-- Differentiable tensor representations supply numeric and fractional dictionaries;
-- 'Given' carries additional evidence already available at the definition site.
data Needs a (c :: Constraint) where
  -- | Numeric tensor arithmetic.
  Numeric :: Needs a (Num a, Num (T.TensorElement a))
  -- | Tensor arithmetic including division.
  Fractional :: Needs a (Fractional a, Fractional (T.TensorElement a))
  -- | An additional constraint supplied by the operation author.
  Given :: (c) => Needs a c
  -- | Combine two capability requests.
  And :: Needs a c -> Needs a d -> Needs a (c, d)

-- | Tensor-only input and output structures accepted by 'customVJP'.
-- Resolved automatically for individual tensors, tuples, and generic records
-- with a 'T.WithTensor' correspondence. Inputs share the operation's element
-- type. Multiple outputs use that same element type; a single tensor result
-- may use another element type.
type Compatible t a input output = (CompatibleInput t a input, View t output, ResultType a (Leaves T.Tensor output))

-- | Tensor input structures accepted by primitive computations.
-- Resolved automatically for tensors, tuples, and generic tensor records.
type CompatibleInput t a input = (View t input, InputType a (Leaves T.Tensor input))

-- | Representations supporting tensor primitives. Ordinary tensors evaluate only
-- the ordinary forward function. Differentiable representations use the cached
-- forward function and register its pullback.
class (DType a) => TensorOps (t :: T.TensorLike) a where
  -- | Embed a staged value without recording any incoming gradient edges.
  literal :: T.Tensor sh a -> t sh a

  -- | Read the staged value without its differentiation history.
  primal :: t sh a -> T.Tensor sh a

  -- | Convert an element expression into a scalar tensor, preserving its history.
  fromElement :: T.Element t a -> t '[] a

  -- | Define an element-valued primitive with a private cache and custom pullback.
  -- Input registration and capability requests follow 'customVJP'. The result
  -- remains an element expression, so it may depend on coordinates local to a
  -- generator without introducing an intermediate scalar tensor.
  -- The pullback receives an element cotangent and returns one tensor cotangent
  -- per input leaf. The cache is used only during differentiation, and the two
  -- forward functions must compute equivalent element expressions.
  customElementVJP ::
    (CompatibleInput t a input) =>
    Needs a c ->
    (input -> T.TensorElement b) ->
    (input -> (T.TensorElement b, cache)) ->
    ((c) => cache -> T.TensorElement b -> input) ->
    T.WithTensor t input ->
    T.Element t b

  -- | Define a primitive over tensor products or generic tensor records.
  -- The two forward functions must return equivalent results. The cached path
  -- may retain arbitrary private values for the pullback; these are not inputs
  -- to differentiation. The pullback receives output cotangents and returns
  -- input cotangents with the same structure and shapes as the original inputs.
  -- All differentiable dependencies must be explicit inputs, not captured variables.
  -- 'Needs' requests dictionaries available during the reverse pass without
  -- imposing them on ordinary tensor callers. Input leaves share the class's
  -- element type. Multiple output leaves use that same element type; a single
  -- tensor output may have another element type.
  --
  -- @
  -- square :: TensorOps t Double => t '[3] Double -> t '[3] Double
  -- square = customVJP Numeric (\x -> x * x)
  --   (\x -> (x * x, x))
  --   (\x seed -> 2 * x * seed)
  -- @
  customVJP ::
    forall input output cache c.
    (Compatible t a input output) =>
    Needs a c ->
    (input -> output) ->
    (input -> (output, cache)) ->
    ((c) => cache -> output -> input) ->
    T.WithTensor t input ->
    T.WithTensor t output

instance (T.DType a) => TensorOps T.Tensor a where
  literal = id
  primal = id
  fromElement (T.TensorElement x) = T.Tensor (A.unit x)
  customElementVJP _ forward _ _ input =
    forward $ rebuild @T.Tensor $ readInputs (inputTree @a) $ flatten @T.Tensor input
  customVJP _ forward _ _ input =
    let original = rebuild @T.Tensor $ readInputs (inputTree @a) $ flatten @T.Tensor input
     in rebuild @T.Tensor $ writeResults (resultTree @a) $ flatten @T.Tensor $ forward original
