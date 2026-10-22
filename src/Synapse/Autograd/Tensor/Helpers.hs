{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Tensor vector-Jacobian products and gradient shape transformations.
module Synapse.Autograd.Tensor.Helpers
  ( tensorUnary,
    tensorBinary,
    reshapeLike,
    expandReduced,
    sumBroadcast,
    shapeOf,
    axesOf,
    reindexGradient,
    singletonAxis,
  )
where

import qualified Data.Array.Accelerate as A
import Data.List (foldl')
import Data.Proxy (Proxy (..))
import GHC.TypeLits (Nat)
import Synapse.Autograd.Tensor.Type (Tensor (..))
import Synapse.Autograd.Type (Backpropagatable, Context, add, liftBinary, liftUnary)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Shapes.Axes (Axes, axes)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendCoordinates, shape, shapeSize, strides)
import qualified Synapse.Tensor.Type as T

-- | Turn a selected axis into a singleton axis list.
singletonAxis :: proxy (axis :: Nat) -> Proxy '[axis]
singletonAxis _ = Proxy

-- | Reflect the statically declared dimensions of a tensor.
shapeOf :: forall sh a. (Shape sh) => T.Tensor sh a -> [Int]
shapeOf _ = shape @sh

-- | Reflect a list of axes selected by a type-level proxy.
axesOf :: forall xs proxy. (Axes xs) => proxy xs -> [Int]
axesOf _ = axes @xs

-- | Read a gradient using a coordinate transformation from the input shape.
reindexGradient :: forall from to a. (Shape from, Shape to, DType a) => ([A.Exp Int] -> [A.Exp Int]) -> T.Tensor from a -> T.Tensor to a -> T.Tensor from a
reindexGradient coordinates (T.Tensor reference) (T.Tensor gradient) = T.Tensor $ A.generate (A.shape reference) $ \ix ->
  let offset = sum (zipWith (*) (coordinates (backendCoordinates @from ix)) (map A.constant (strides (shape @to))))
   in A.flatten gradient A.! A.index1 offset

-- | Register a tensor operation and its pullback to the input shape.
tensorUnary :: (Context s, Backpropagatable (T.Tensor from a)) => (T.Tensor from a -> (T.Tensor to b, T.Tensor to b -> T.Tensor from a)) -> Tensor s from a -> Tensor s to b
tensorUnary rule (Tensor x) = Tensor (liftUnary add rule x)

-- | Register a tensor operation with independently accumulated input gradients.
tensorBinary :: (Context s, Backpropagatable (T.Tensor lhs a), Backpropagatable (T.Tensor rhs b)) => (T.Tensor lhs a -> T.Tensor rhs b -> (T.Tensor out c, T.Tensor out c -> (T.Tensor lhs a, T.Tensor rhs b))) -> Tensor s lhs a -> Tensor s rhs b -> Tensor s out c
tensorBinary rule (Tensor x) (Tensor y) = Tensor (liftBinary add add rule x y)

-- | Restore the reference shape without requiring a type-level proof in the reverse direction.
reshapeLike :: (Shape from, Shape to, DType a) => T.Tensor from a -> T.Tensor to a -> T.Tensor from a
reshapeLike (T.Tensor reference) (T.Tensor gradient) = T.Tensor (A.reshape (A.shape reference) gradient)

-- | Repeat a reduced gradient over the axes removed by the forward reduction.
expandReduced :: forall from to a. (Shape from, Shape to, DType a) => [Int] -> T.Tensor from a -> T.Tensor to a -> T.Tensor from a
expandReduced selected (T.Tensor reference) (T.Tensor gradient) = T.Tensor $ A.generate (A.shape reference) $ \ix ->
  let coordinates = backendCoordinates @from ix
      remaining = [coordinate | (axis, coordinate) <- zip [0 ..] coordinates, axis `notElem` selected]
      offset = sum (zipWith (*) remaining (map A.constant (strides (shape @to))))
   in A.flatten gradient A.! A.index1 offset

-- | Sum all broadcast copies into the original shape using one segmented reduction.
sumBroadcast :: forall from to a. (Shape from, Shape to, DType a, Num (T.TensorElement a)) => T.Tensor from a -> T.Tensor to a -> T.Tensor from a
sumBroadcast (T.Tensor reference) (T.Tensor gradient) = T.Tensor $ A.reshape (A.shape reference) $ A.fold plus zero contributions
  where
    fromDims = shape @from
    toDims = shape @to
    padding = length toDims - length fromDims
    padded = replicate padding 1 ++ fromDims
    reduced = [axis | (axis, (source, target)) <- zip [0 ..] (zip padded toDims), axis < padding || source /= target]
    reducedDims = [dim | (axis, dim) <- zip [0 ..] toDims, axis `elem` reduced]
    width = product reducedDims
    zero = T.unTensorElement (0 :: T.TensorElement a)
    plus :: A.Exp a -> A.Exp a -> A.Exp a
    plus x y = T.unTensorElement (T.TensorElement x + T.TensorElement y)
    contributions = A.generate (A.constant (A.Z A.:. shapeSize @from A.:. width)) $ \ix ->
      let linear = A.indexHead (A.indexTail ix)
          inner = A.indexHead ix
          coordinates dims value = zipWith (\dim stride -> value `A.quot` A.constant stride `A.rem` A.constant dim) dims (strides dims)
          original = replicate padding 0 ++ coordinates fromDims linear
          copies = coordinates reducedDims inner
          inject [] _ = []
          inject ((axis, coordinate) : rest) values
            | axis `elem` reduced = case values of
                v : vs -> v : inject rest vs
                [] -> error "Synapse.Autograd.sumBroadcast: inconsistent broadcast rank"
            | otherwise = coordinate : inject rest values
          target = inject (zip [0 ..] original) copies
          offset = foldl' (+) 0 (zipWith (*) target (map A.constant (strides toDims)))
       in A.flatten gradient A.! A.index1 offset
