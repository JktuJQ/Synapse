{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}

-- | Shape transformations used by tensor pullbacks.
module Synapse.Autograd.Tensor.Ops.Primitives.Shape
  ( reshape,
    broadcast,
    permuteAxes,
    transpose,
    concat,
    stack,
    expandReduced,
    shapeOf,
    axesOf,
    singletonAxis,
    joinMatrixChunks,
  )
where

import qualified Data.Array.Accelerate as A
import Data.List (foldl')
import Data.Proxy (Proxy (..))
import GHC.TypeLits (KnownNat, Nat, natVal, type (*))
import Synapse.Tensor.Shapes.Axes (Axes, axes)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendCoordinates, shape, shapeSize, strides)
import Synapse.Tensor.Type (DType, Tensor (..), TensorElement (..))
import qualified Synapse.Tensor.Type as T
import Prelude hiding (concat)

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

-- | Restore the original shape after a reshape or singleton-axis transformation.
reshape :: (Shape from, Shape to, DType a) => T.Tensor from a -> T.Tensor to a -> T.Tensor to a -> T.Tensor from a
reshape input _ = reshapeLike input

-- | Sum gradient contributions from every broadcast copy.
broadcast :: (Shape from, Shape to, DType a, Num (T.TensorElement a)) => T.Tensor from a -> T.Tensor to a -> T.Tensor to a -> T.Tensor from a
broadcast input _ = sumBroadcast input

-- | Invert a validated permutation of the input axes.
permuteAxes :: forall xs from to a proxy. (Axes xs, Shape from, Shape to, DType a) => proxy xs -> T.Tensor from a -> T.Tensor to a -> T.Tensor to a -> T.Tensor from a
permuteAxes selected input _ = reindexGradient (\coordinates -> map (coordinates !!) (axesOf selected)) input

-- | Invert the exchange of the last two axes.
transpose :: (Shape from, Shape to, DType a) => T.Tensor from a -> T.Tensor to a -> T.Tensor to a -> T.Tensor from a
transpose input _ =
  let rank = length (shapeOf input)
      order = [0 .. rank - 3] ++ [rank - 1, rank - 2]
   in reindexGradient (\coordinates -> map (coordinates !!) order) input

-- | Split a concatenated gradient at the original left input's axis extent.
concat :: (KnownNat axis, Shape lhs, Shape rhs, Shape out, DType a) => proxy axis -> T.Tensor lhs a -> T.Tensor rhs a -> T.Tensor out a -> T.Tensor out a -> (T.Tensor lhs a, T.Tensor rhs a)
concat selected left right _ gradient =
  let axis = fromInteger (toInteger (natVal selected))
      offset = shapeOf left !! axis
      shift = zipWith (\i coordinate -> if i == axis then coordinate + A.constant offset else coordinate) [0 ..]
   in (reindexGradient id left gradient, reindexGradient shift right gradient)

-- | Select each input's position along the inserted stack axis.
stack :: (KnownNat axis, Shape sh, Shape out, DType a) => proxy axis -> T.Tensor sh a -> T.Tensor sh a -> T.Tensor out a -> T.Tensor out a -> (T.Tensor sh a, T.Tensor sh a)
stack selected left right _ gradient =
  let axis = fromInteger (toInteger (natVal selected))
      insert value coordinates = take axis coordinates ++ [value] ++ drop axis coordinates
   in (reindexGradient (insert 0) left gradient, reindexGradient (insert 1) right gradient)

-- | Join gate cotangents in their original packed order.
-- The list must contain exactly @gates@ tensors.
joinMatrixChunks :: forall gates hidden batch a. (KnownNat gates, KnownNat hidden, KnownNat batch, DType a, Num (TensorElement a)) => [Tensor '[batch, hidden] a] -> Tensor '[batch, gates * hidden] a
joinMatrixChunks parts = Tensor $ A.generate (A.constant $ A.Z A.:. batch A.:. (count * width)) $ \ix ->
  let row = A.indexHead $ A.indexTail ix
      column = A.indexHead ix
      select :: [Tensor '[batch, hidden] a] -> Int -> A.Exp a
      select [] _ = unTensorElement (0 :: TensorElement a)
      select (Tensor part : rest) offset =
        A.cond
          (column A.< A.constant (offset + width))
          (part A.! A.index2 row (column - A.constant offset))
          (select rest $ offset + width)
   in select parts 0
  where
    batch = fromIntegral $ natVal $ Proxy @batch
    count = fromIntegral $ natVal $ Proxy @gates
    width = fromIntegral $ natVal $ Proxy @hidden
