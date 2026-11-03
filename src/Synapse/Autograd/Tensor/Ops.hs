{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Register independently accumulated tensor leaves and structured results.
module Synapse.Autograd.Tensor.Ops () where

import qualified Data.Array.Accelerate as A
import Data.Functor.Identity (Identity (..))
import qualified Numeric.Backprop.Explicit as BPE
import Numeric.Backprop.Op (Rec (..))
import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
import Synapse.Autograd.Variable (Context, Var (..), add, liftUnary, variableValue)
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Ops (Compatible, CompatibleInput, Needs (..), TensorOps (..))
import Synapse.Tensor.Ops.Structure
import Synapse.Tensor.Type (DType)
import qualified Synapse.Tensor.Type as T

withNeeds :: (Fractional a, Fractional (T.TensorElement a)) => Needs a c -> ((c) => r) -> r
withNeeds Numeric value = value
withNeeds Fractional value = value
withNeeds Given value = value
withNeeds (And left right) value = withNeeds left $ withNeeds right value

type family xs ++ ys where
  '[] ++ ys = ys
  (x ': xs) ++ ys = x ': (xs ++ ys)

append :: Rec f xs -> Rec f ys -> Rec f (xs ++ ys)
append RNil ys = ys
append (x :& xs) ys = x :& append xs ys

split :: Rec g xs -> Rec f (xs ++ ys) -> (Rec f xs, Rec f ys)
split RNil values = (RNil, values)
split (_ :& xs) (v :& values) = let (a, b) = split xs values in (v :& a, b)

data Inputs s input where
  Inputs :: Rec BPE.AddFunc xs -> Rec (BPE.BVar s) xs -> (Rec Identity xs -> input) -> (input -> Rec Identity xs) -> Inputs s input

registerInputs :: forall s a input. (T.DType a, Num a, Num (T.TensorElement a)) => InputTree a input -> Wrapped (Tensor s) input -> Inputs s input
registerInputs InputTensor (Tensor (Var value)) = Inputs (BPE.AF (+) :& RNil) (value :& RNil) (\(Identity x :& RNil) -> x) (\x -> Identity x :& RNil)
registerInputs (InputPair l r) (a, b) = case (registerInputs l a, registerInputs r b) of
  (Inputs la lv lr lp, Inputs ra rv rr rp) -> Inputs (append la ra) (append lv rv) (\values -> let (ls, rs) = split la values in (lr ls, rr rs)) (\(x, y) -> append (lp x) (rp y))

wrapResults :: forall s a output. (Context s, T.DType a, Num a, Num (T.TensorElement a)) => Result a output -> BPE.BVar s output -> Wrapped (Tensor s) output
wrapResults Single = Tensor . Var
wrapResults (Product tree) = wrapProduct tree

-- | Project each output leaf from one shared tape node. Contributions from
-- different projections accumulate before the primitive pullback is evaluated.
wrapProduct :: (Context s, T.DType a, Num a, Num (T.TensorElement a)) => InputTree a output -> BPE.BVar s output -> Wrapped (Tensor s) output
wrapProduct InputTensor value = Tensor $ Var value
wrapProduct tree@(InputPair l r) value =
  let left = BPE.liftOp1 (BPE.AF $ addInput tree) (BPE.op1 $ \(x, _) -> (x, (,zeroInput r))) value
      right = BPE.liftOp1 (BPE.AF $ addInput tree) (BPE.op1 $ \(_, y) -> (y, (zeroInput l,))) value
   in (wrapProduct l left, wrapProduct r right)

-- | Register tensor leaves independently, retaining arbitrary private residuals.
-- The output may be a tensor product or a coordinate-local element expression.
registerPrimitive ::
  forall s a input output cache c.
  (Context s, DType a, Fractional a, Fractional (T.TensorElement a), CompatibleInput (Tensor s) a input) =>
  Needs a c ->
  (input -> (output, cache)) ->
  ((c) => cache -> output -> input) ->
  T.WithTensor (Tensor s) input ->
  BPE.BVar s output
registerPrimitive needs forward backward input =
  case registerInputs @s (inputTree @a @(Leaves T.Tensor input)) (flatten @(Tensor s) input) of
    Inputs additions variables unpack pack ->
      BPE.liftOp
        additions
        ( BPE.Op $ \arguments ->
            let original = rebuild @T.Tensor $ unpack arguments
                (output, cache) = forward original
                pullback seed = withNeeds needs $ pack $ flatten @T.Tensor $ backward cache seed
             in (output, pullback)
        )
        variables

instance (Context s, DType a, Fractional a, Fractional (T.TensorElement a)) => TensorOps (Tensor s) a where
  literal = Tensor . Var . BPE.auto
  primal = variableValue . unTensor
  fromElement (TensorElement x) = Tensor $ liftUnary add (\(T.TensorElement value) -> (T.Tensor (A.unit value), \(T.Tensor seed) -> T.TensorElement (A.the seed))) x
  customElementVJP needs _ forward backward = TensorElement . Var . registerPrimitive needs forward backward
  customVJP :: forall input output cache c. (Compatible (Tensor s) a input output) => Needs a c -> (input -> output) -> (input -> (output, cache)) -> ((c) => cache -> output -> input) -> T.WithTensor (Tensor s) input -> T.WithTensor (Tensor s) output
  customVJP needs _ forward backward input =
    let cached arguments = let (output, cache) = forward arguments in (flatten @T.Tensor output, cache)
        result = registerPrimitive needs cached (\cache seed -> backward cache $ rebuild @T.Tensor seed) input
     in rebuild @(Tensor s) $ wrapResults @s (resultTree @a @(Leaves T.Tensor output)) result
