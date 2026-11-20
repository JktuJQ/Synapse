{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | TensorOps instance for differentiable tensors.
module Synapse.Autograd.Tensor.Ops (unaryVJP, binaryVJP) where

import Data.Functor.Identity (Identity (..))
import Data.Proxy (Proxy)
import Numeric.Backprop.Op (Rec (..))
import qualified Synapse.Tensor.Representation as T

import Synapse.Autograd.Record (append, split)
import Synapse.Autograd.Representation (Backpropagatable, Context, add)
import Synapse.Autograd.Tensor.Representation (Tensor (..), TensorElement (..))
import Synapse.Tensor.Ops.Constraints (FractionalDType, NumDType)
import Synapse.Tensor.Ops.Definition
    ( Compatible
    , CompatibleInput
    , InputTree (..)
    , InputType (..)
    , Result (..)
    , ResultType (..)
    , TensorOps (..)
    , Wrapped
    , addInput
    , resultFields
    , resultValues
    , zeroInput
    )
import Synapse.Tensor.Structure (TensorProduct, flatten, rebuild)
import Synapse.Tensor.Structure.Product (ProductValues)

import qualified Data.Array.Accelerate as A
import qualified Numeric.Backprop.Explicit as BPE
import qualified Numeric.Backprop.Internal as BPI

import qualified Synapse.Tensor.Structure as T

-- | Existing input variables, accumulation rules, and conversions for one primitive.
data PrimitiveInputs s input where
    PrimitiveInputs
        :: Rec BPE.AddFunc xs
        -> Rec (BPE.BVar s) xs
        -> (Rec Identity xs -> input)
        -> (input -> Rec Identity xs)
        -> PrimitiveInputs s input

-- | Collect existing tensor variables and their gradient accumulation rules.
preparePrimitiveInputs
    :: forall s a input
     . NumDType a
    => InputTree a input
    -> Wrapped (Tensor s) input
    -> PrimitiveInputs s input
preparePrimitiveInputs InputTensor (T.TensorField path (Tensor value)) =
    PrimitiveInputs
        (BPE.AF (+) :& RNil)
        (value :& RNil)
        (\(Identity x :& RNil) -> T.TensorField path x)
        (\(T.TensorField _ x) -> Identity x :& RNil)
preparePrimitiveInputs (InputPair l r) (a, b) =
    case (preparePrimitiveInputs l a, preparePrimitiveInputs r b) of
        (PrimitiveInputs la lv lr lp, PrimitiveInputs ra rv rr rp) ->
            PrimitiveInputs
                (append la ra)
                (append lv rv)
                (\values -> let (ls, rs) = split la values in (lr ls, rr rs))
                (\(x, y) -> append (lp x) (rp y))

-- | Wrap a single result or project a product into differentiable tensor fields.
wrapResults
    :: forall s a output
     . (Context s, NumDType a)
    => Result a output
    -> BPE.BVar s (ProductValues output)
    -> Wrapped (Tensor s) output
wrapResults Single = T.TensorField [] . Tensor
wrapResults (Product tree) = wrapProduct tree

{- | Project each output tensor from one shared tape node.
Contributions from different projections accumulate before the primitive pullback is evaluated.
-}
wrapProduct
    :: (Context s, NumDType a)
    => InputTree a output
    -> BPE.BVar s (ProductValues output)
    -> Wrapped (Tensor s) output
wrapProduct InputTensor value = T.TensorField [] (Tensor value)
wrapProduct tree@(InputPair l r) value =
    let left = BPE.liftOp1 (BPE.AF $ addInput tree) (BPE.op1 $ \(x, _) -> (x, (,zeroInput r))) value
        right = BPE.liftOp1 (BPE.AF $ addInput tree) (BPE.op1 $ \(_, y) -> (y, (zeroInput l,))) value
     in (wrapProduct l left, wrapProduct r right)

-- | Register input tensors and attach the cached forward calculation and its pullback.
registerPrimitive
    :: forall s a input output cache
     . (Context s, FractionalDType a, CompatibleInput (Tensor s) a input)
    => (input -> (output, cache))
    -> (cache -> output -> input)
    -> T.WithTensor (Tensor s) input
    -> BPE.BVar s output
registerPrimitive forward backward input =
    case preparePrimitiveInputs @s
        (inputTree @a @(TensorProduct T.Tensor input))
        (flatten @input @(Tensor s) input) of
        PrimitiveInputs additions variables unpack pack ->
            BPE.liftOp
                additions
                ( BPE.Op $ \arguments ->
                    let original = rebuild @input @T.Tensor $ unpack arguments
                        (output, cache) = forward original
                        pullback seed = pack $ flatten @input @T.Tensor $ backward cache seed
                     in (output, pullback)
                )
                variables

-- | Lift a unary operation, retaining its input and output for the pullback.
unaryVJP
    :: (Context s, Backpropagatable a)
    => (a -> a)
    -> (a -> a -> a -> a)
    -> BPE.BVar s a
    -> BPE.BVar s a
unaryVJP forward backward = BPE.liftOp1 (BPE.AF add) $ BPE.op1 $ \x ->
    let output = forward x
     in (output, backward x output)

-- | Lift a binary operation with one gradient contribution per input.
binaryVJP
    :: (Context s, Backpropagatable a)
    => (a -> a -> a)
    -> (a -> a -> a -> a -> (a, a))
    -> BPE.BVar s a
    -> BPE.BVar s a
    -> BPE.BVar s a
binaryVJP forward backward = BPE.liftOp2 (BPE.AF add) (BPE.AF add) $ BPE.op2 $ \x y ->
    let output = forward x y
     in (output, backward x y output)

instance (Context s, FractionalDType a) => TensorOps (Tensor s) a where
    type Element (Tensor s) = TensorElement s
    type BackwardConstraints (Tensor s) c = c
    fromTensor = Tensor . BPE.auto
    toTensor = BPI._bvVal . unTensor
    fromElement (TensorElement x) =
        Tensor $
            BPE.liftOp1
                (BPE.AF add)
                ( BPE.op1 $ \(T.TensorElement value) ->
                    (T.Tensor (A.unit value), \(T.Tensor seed) -> T.TensorElement (A.the seed))
                )
                x
    customElementVJP _ _ forward backward = TensorElement . registerPrimitive forward backward
    customVJP
        :: forall input output cache c
         . (Compatible (Tensor s) a input output, BackwardConstraints (Tensor s) c)
        => Proxy c
        -> (input -> output)
        -> (input -> (output, cache))
        -> (c => cache -> output -> input)
        -> T.WithTensor (Tensor s) input
        -> T.WithTensor (Tensor s) output
    customVJP _ _ forward backward input =
        let outputTree = resultTree @a @(TensorProduct T.Tensor output)
            cached arguments =
                let (output, cache) = forward arguments
                 in ( resultValues outputTree (flatten @output @T.Tensor output)
                    , cache
                    )
            result =
                registerPrimitive
                    cached
                    ( \cache seed ->
                        backward cache $
                            rebuild @output @T.Tensor $
                                resultFields outputTree seed
                    )
                    input
         in rebuild @output @(Tensor s) $
                wrapResults @s outputTree result
