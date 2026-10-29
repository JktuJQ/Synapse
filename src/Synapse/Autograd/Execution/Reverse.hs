{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

-- | Reverse-mode execution with independent inputs on the <https://hackage.haskell.org/package/backprop backprop> tape.
module Synapse.Autograd.Execution.Reverse
  ( pullback,
  )
where

import Data.Functor.Identity (Identity (..))
import qualified Numeric.Backprop.Explicit as BPE
import Numeric.Backprop.Op (Rec (..))
import Synapse.Autograd.Execution.Variable (Unconstrained, ValueTree (..), Variables, add, pack)
import qualified Synapse.Autograd.Tensor.Type as Tensor
import Synapse.Autograd.Variable (Context)
import qualified Synapse.Autograd.Variable as AG

-- | Type-level concatenation of independently registered inputs.
type family xs :++ ys where
  '[] :++ ys = ys
  (x ': xs) :++ ys = x ': (xs :++ ys)

-- | Heterogeneous tape inputs with conversions to the public representation.
data Inputs a where
  Inputs ::
    Rec BPE.ZeroFunc xs ->
    Rec BPE.AddFunc xs ->
    Rec Identity xs ->
    (Rec Identity xs -> a) ->
    (forall s. Rec (BPE.BVar s) xs -> Variables s a) ->
    Inputs a

-- | Append input records without converting the stored values.
append :: Rec f xs -> Rec f ys -> Rec f (xs :++ ys)
append RNil ys = ys
append (x :& xs) ys = x :& append xs ys

-- | Split a record using the existing left input record as its structural witness.
split :: Rec g xs -> Rec f (xs :++ ys) -> (Rec f xs, Rec f ys)
split RNil values = (RNil, values)
split (_ :& rest) (value :& values) =
  let (left, right) = split rest values
   in (value :& left, right)

-- | Register each leaf as its own tape input.
inputs :: ValueTree Unconstrained a -> a -> Inputs a
inputs TensorValues value =
  Inputs
    (BPE.ZF AG.zero :& RNil)
    (BPE.AF AG.add :& RNil)
    (Identity value :& RNil)
    (\(Identity x :& RNil) -> x)
    (\(x :& RNil) -> Tensor.Tensor (AG.Var x))
inputs (PairValues left right) (x, y) =
  case (inputs left x, inputs right y) of
    (Inputs leftZeros leftAdds leftValues rebuildLeft wrapLeft, Inputs rightZeros rightAdds rightValues rebuildRight wrapRight) ->
      Inputs
        (append leftZeros rightZeros)
        (append leftAdds rightAdds)
        (append leftValues rightValues)
        (\arguments -> let (ls, rs) = split leftZeros arguments in (rebuildLeft ls, rebuildRight rs))
        (\(arguments :: Rec (BPE.BVar s) leaves) -> let (ls, rs) = split leftZeros arguments in (wrapLeft @s ls, wrapRight @s rs))

-- | Run a reverse pass while registering each leaf as an independent input.
pullback ::
  ValueTree Unconstrained a ->
  ValueTree Unconstrained b ->
  (forall s. (Context s) => Variables s a -> Variables s b) ->
  a ->
  (b, b -> a)
pullback inputStructure resultStructure function input = case inputs inputStructure input of
  Inputs zeroFunctions additions values rebuild variables ->
    let (value, backward) =
          BPE.backpropWithN
            zeroFunctions
            (\(arguments :: Rec (BPE.BVar s) leaves) -> outputRoot resultStructure zeroFunctions additions arguments (pack @s resultStructure (function @s (variables arguments))))
            values
     in (value, rebuild . backward)

-- | Anchor the requested output to a live input so that it becomes the final tape node.
-- The extra edge contributes a reference-shaped zero directly to the input.
outputRoot :: (Context s) => ValueTree Unconstrained b -> Rec BPE.ZeroFunc xs -> Rec BPE.AddFunc xs -> Rec (BPE.BVar s) xs -> BPE.BVar s b -> BPE.BVar s b
outputRoot result zeros additions variables value = case (zeros, additions, variables) of
  (zero :& _, addition :& _, variable :& _) ->
    BPE.liftOp2
      (BPE.AF (add result))
      addition
      (BPE.op2 (\output reference -> (output, (,BPE.runZF zero reference))))
      value
      variable
  (RNil, RNil, RNil) -> value
