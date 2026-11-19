{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-orphans #-}

-- | Comparisons and conditional selection for differentiable values.
module Synapse.Autograd.Tensor.Ops.Logic () where

import Synapse.Autograd.Representation (Context)
import Synapse.Autograd.Tensor.Ops (binaryVJP)
import Synapse.Autograd.Tensor.Representation (Tensor (..), TensorElement (..))
import Synapse.Tensor.Ops.Constraints (EqDType, FractionalDType, NumDType, OrdDType)
import qualified Synapse.Tensor.Representation as T
import Synapse.Tensor.Shapes.Dimensions (Shape)

import qualified Numeric.Backprop.Internal as BPI

import qualified Synapse.Autograd.Tensor.Ops.Pullback.Logic as Pullback
import qualified Synapse.Tensor.Ops.Logic as E

instance
    (Context s, Shape sh, FractionalDType a)
    => E.Conditional (Tensor s sh a)
    where
    type Condition (Tensor s sh a) = T.Tensor sh Bool
    type Predicate (Tensor s sh a) = T.Tensor '[] Bool
    select mask (Tensor x) (Tensor y) = Tensor $ binaryVJP (E.select mask) (Pullback.select mask) x y
    cond flag (Tensor x) (Tensor y) = Tensor $ binaryVJP (E.cond flag) (Pullback.cond flag) x y
instance (Context s, NumDType a) => E.Conditional (TensorElement s a) where
    type Condition (TensorElement s a) = T.TensorElement Bool
    type Predicate (TensorElement s a) = T.TensorElement Bool
    select mask (TensorElement x) (TensorElement y) =
        TensorElement (binaryVJP (E.select mask) (Pullback.select mask) x y)
    cond flag (TensorElement x) (TensorElement y) =
        TensorElement (binaryVJP (E.cond flag) (Pullback.cond flag) x y)
instance E.Eq (T.Tensor sh a) => E.Eq (Tensor s sh a) where
    type Comparison (Tensor s sh a) = T.Tensor sh Bool
    Tensor x == Tensor y = BPI._bvVal x E.== BPI._bvVal y
    Tensor x /= Tensor y = BPI._bvVal x E./= BPI._bvVal y
instance EqDType a => E.Eq (TensorElement s a) where
    type Comparison (TensorElement s a) = T.TensorElement Bool
    TensorElement x == TensorElement y = BPI._bvVal x E.== BPI._bvVal y
    TensorElement x /= TensorElement y = BPI._bvVal x E./= BPI._bvVal y
instance
    (Context s, Shape sh, FractionalDType a, OrdDType a)
    => E.Ord (Tensor s sh a)
    where
    type OrderingResult (Tensor s sh a) = T.Tensor sh Ordering
    compare (Tensor x) (Tensor y) = E.compare (BPI._bvVal x) (BPI._bvVal y)
    Tensor x < Tensor y = BPI._bvVal x E.< BPI._bvVal y
    Tensor x <= Tensor y = BPI._bvVal x E.<= BPI._bvVal y
    Tensor x > Tensor y = BPI._bvVal x E.> BPI._bvVal y
    Tensor x >= Tensor y = BPI._bvVal x E.>= BPI._bvVal y
    min (Tensor x) (Tensor y) = Tensor $ binaryVJP E.min (Pullback.extremum (E.>)) x y
    max (Tensor x) (Tensor y) = Tensor $ binaryVJP E.max (Pullback.extremum (E.<)) x y
instance
    (Context s, FractionalDType a, OrdDType a)
    => E.Ord (TensorElement s a)
    where
    type OrderingResult (TensorElement s a) = T.TensorElement Ordering
    compare (TensorElement x) (TensorElement y) = E.compare (BPI._bvVal x) (BPI._bvVal y)
    TensorElement x < TensorElement y = BPI._bvVal x E.< BPI._bvVal y
    TensorElement x <= TensorElement y = BPI._bvVal x E.<= BPI._bvVal y
    TensorElement x > TensorElement y = BPI._bvVal x E.> BPI._bvVal y
    TensorElement x >= TensorElement y = BPI._bvVal x E.>= BPI._bvVal y
    min (TensorElement x) (TensorElement y) =
        TensorElement (binaryVJP E.min (Pullback.extremum (E.>)) x y)
    max (TensorElement x) (TensorElement y) =
        TensorElement (binaryVJP E.max (Pullback.extremum (E.<)) x y)
