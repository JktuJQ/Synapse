{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}

-- | Internal product witnesses for tensor primitives.
module Synapse.Tensor.Ops.Structure where

import Data.Bifunctor (bimap)
import Synapse.Tensor.Fields (Flatten (..), Leaves)
import Synapse.Tensor.Ops.Elementwise ()
import Synapse.Tensor.Ops.Primitives.Math ()
import Synapse.Tensor.Shapes.Dimensions (Shape)
import qualified Synapse.Tensor.Type as T

type family Wrapped t a where
  Wrapped t (T.Tensor sh a) = t sh a
  Wrapped t (a, b) = (Wrapped t a, Wrapped t b)

class (Flatten T.Tensor a, Flatten t (T.WithTensor t a), Leaves t (T.WithTensor t a) ~ Wrapped t (Leaves T.Tensor a)) => View t a

instance (Flatten T.Tensor a, Flatten t (T.WithTensor t a), Leaves t (T.WithTensor t a) ~ Wrapped t (Leaves T.Tensor a)) => View t a

-- All registered input leaves share the TensorOps scalar type.
data InputTree a input where
  InputTensor :: (Shape sh) => InputTree a (T.Tensor sh a)
  InputPair :: InputTree a l -> InputTree a r -> InputTree a (l, r)

class InputType a input | input -> a where inputTree :: InputTree a input

instance (Shape sh) => InputType a (T.Tensor sh a) where inputTree = InputTensor

instance (InputType a l, InputType a r) => InputType a (l, r) where inputTree = InputPair (inputTree @a @l) (inputTree @a @r)

readInputs :: InputTree a input -> Wrapped T.Tensor input -> input
readInputs InputTensor = id
readInputs (InputPair l r) = bimap (readInputs l) (readInputs r)

zeroInput :: (T.DType a, Num a, Num (T.TensorElement a)) => InputTree a input -> input
zeroInput InputTensor = 0
zeroInput (InputPair l r) = (zeroInput l, zeroInput r)

addInput :: (T.DType a, Num a, Num (T.TensorElement a)) => InputTree a input -> input -> input -> input
addInput InputTensor = (+)
addInput (InputPair l r) = \(a, b) (c, d) -> (addInput l a c, addInput r b d)

-- | Single results may change element type without requiring numeric output evidence.
data Result a output where
  Single :: Result a (T.Tensor sh b)
  Product :: InputTree a output -> Result a output

class ResultType a output where resultTree :: Result a output

instance ResultType a (T.Tensor sh b) where resultTree = Single

instance (InputType a l, InputType a r) => ResultType a (l, r) where resultTree = Product $ InputPair (inputTree @a @l) (inputTree @a @r)

writeResults :: Result a output -> output -> Wrapped T.Tensor output
writeResults Single = id
writeResults (Product tree) = writeProduct tree

writeProduct :: InputTree a output -> output -> Wrapped T.Tensor output
writeProduct InputTensor = id
writeProduct (InputPair l r) = bimap (writeProduct l) (writeProduct r)
