{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Element access and slices with gradients restored to the source shape.
module Synapse.Autograd.Tensor.Ops.Index () where

import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy)
import qualified Numeric.Backprop.Explicit as BP
import Synapse.Autograd.Tensor.Helpers (shapeOf, tensorUnary)
import Synapse.Autograd.Tensor.Type (Tensor (..), TensorElement (..))
import Synapse.Autograd.Type (Context, Var (..), add, liftUnary)
import Synapse.Tensor.DType (DType)
import Synapse.Tensor.Ops.Index (Coordinates (..), Indexable (..), Indexing (..), Slicing (..))
import Synapse.Tensor.Shapes.Dimensions (Shape, backendCoordinates, strides)
import Synapse.Tensor.Shapes.Index (CanIndex, Index, Slice, SliceAxis (..), coordinates, sliceAxes)
import qualified Synapse.Tensor.Type as T

-- | Reflect a statically checked complete index.
indexCoordinates :: forall ix proxy. (Index ix) => proxy ix -> [Int]
indexCoordinates _ = coordinates @ix

-- | Reflect the descriptors of a statically checked slice.
sliceDescriptors :: forall spec proxy. (Slice spec) => proxy spec -> [SliceAxis Int]
sliceDescriptors _ = sliceAxes @spec

instance Coordinates (TensorElement s) where
  coordinateOp selected = TensorElement . Var . BP.auto . coordinateOp selected

instance (Context s, DType a, Num a, Num (T.TensorElement a)) => Indexing (Tensor s) a where
  indexOp selected (Tensor x) =
    TensorElement $
      liftUnary
        add
        ( \reference@(T.Tensor input) ->
            let offset = sum (zipWith (*) (indexCoordinates selected) (strides (shapeOf reference)))
             in ( indexOp selected reference,
                  \(T.TensorElement gradient) -> T.Tensor $ A.generate (A.shape input) $ \ix ->
                    A.cond (A.toIndex (A.shape input) ix A.== A.constant offset) gradient (T.unTensorElement 0)
                )
        )
        x

instance (Context s, DType a, Num a, Num (T.TensorElement a)) => Slicing (Tensor s) a where
  sliceOp selected = tensorUnary $ \x ->
    (sliceOp selected x, slicePullback (sliceDescriptors selected) x)

instance (Context s, Shape sh, DType a, Num a, Num (T.TensorElement a), Index ix, CanIndex ix sh) => Indexable (Tensor s sh a) (Proxy ix) (TensorElement s a) where
  tensor !! selected = indexOp selected tensor

instance (Context s, Shape sh, DType a, Num a, Num (T.TensorElement a)) => Indexable (Tensor s sh a) (T.TensorIndex sh) (TensorElement s a) where
  Tensor x !! T.TensorIndex selected =
    TensorElement $
      liftUnary
        add
        ( \(T.Tensor input) ->
            ( T.TensorElement (input A.! selected),
              \(T.TensorElement gradient) -> T.Tensor $ A.generate (A.shape input) $ \ix ->
                A.cond (A.toIndex (A.shape input) ix A.== A.toIndex (A.shape input) selected) gradient (T.unTensorElement 0)
            )
        )
        x

-- | Map source coordinates back into a slice, filling unselected positions with zero.
slicePullback :: forall from to a. (Shape from, Shape to, DType a, Num (T.TensorElement a)) => [SliceAxis Int] -> T.Tensor from a -> T.Tensor to a -> T.Tensor from a
slicePullback descriptors (T.Tensor reference) (T.Tensor gradient) = T.Tensor $ A.generate (A.shape reference) $ \ix ->
  let (inside, selected) = select descriptors (backendCoordinates @from ix)
      offset = sum (zipWith (*) selected (map A.constant (strides (shapeOf (T.Tensor gradient :: T.Tensor to a)))))
   in A.cond inside (A.flatten gradient A.! A.index1 offset) (T.unTensorElement (0 :: T.TensorElement a))
  where
    select [] [] = (A.constant True, [])
    select (descriptor : rest) (coordinate : remaining) =
      let (inside, selected) = select rest remaining
          range begin stop step reversed =
            let delta = coordinate - A.constant begin
                count = if stop <= begin then 0 else 1 + (stop - begin - 1) `quot` step
                position = delta `A.quot` A.constant step
                output = if reversed then A.constant (count - 1) - position else position
                valid = coordinate A.>= A.constant begin A.&& coordinate A.< A.constant stop A.&& delta `A.rem` A.constant step A.== 0
             in (inside A.&& valid, output : selected)
       in case descriptor of
            All -> (inside, coordinate : selected)
            At fixed -> (inside A.&& coordinate A.== A.constant fixed, selected)
            Range begin stop step -> range begin stop step False
            RangeReverse begin stop step -> range begin stop step True
    select _ _ = error "Synapse.Autograd.slice: inconsistent slice rank"
