{-# LANGUAGE ImpredicativeTypes #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

{- | Capabilities of element types and tensor representations.
Quantified aliases require @QuantifiedConstraints@; combined aliases also require @ImpredicativeTypes@.
-}
module Synapse.Tensor.Ops.Constraints
    ( -- * Element capabilities
      EqDType
    , OrdDType
    , NumDType
    , RealDType
    , FractionalDType
    , IntegralDType
    , FloatingDType
    , RealFracDType
    , RealFloatDType

      -- * Capabilities across all shapes
    , EqTensors
    , OrdTensors
    , NumTensors
    , FractionalTensors
    , IntegralTensors
    , FloatingTensors
    , RealFracTensors
    , RealFloatTensors
    , ConditionalTensors

      -- * Combined capabilities
    , NumTensorOps
    , FractionalTensorOps
    , IntegralTensorOps
    , FloatingTensorOps
    , RealFracTensorOps
    , RealFloatTensorOps
    ) where

import Data.Kind (Constraint)

import Synapse.Tensor.Ops.Definition (TensorOps)
import Synapse.Tensor.Representation (DType, TensorElement)
import Synapse.Tensor.Shapes.Dimensions (Shape)

import qualified Synapse.Tensor.Ops.Logic as E
import qualified Synapse.Tensor.Ops.Numeric as E

-- | Equality comparisons on host values and element expressions.
type EqDType a = (DType a, Eq a, E.Eq (TensorElement a))

-- | Ordered comparisons on host values and element expressions.
type OrdDType a = (EqDType a, Ord a, E.Ord (TensorElement a))

-- | Numeric operations on host values and element expressions.
type NumDType a = (DType a, Num a, Num (TensorElement a))

-- | Ordered arithmetic and conversion of host values to rational numbers.
type RealDType a = (NumDType a, OrdDType a, Real a)

-- | Fractional operations on host values and element expressions.
type FractionalDType a = (NumDType a, Fractional a, Fractional (TensorElement a))

-- | Integer arithmetic on host values and element expressions.
type IntegralDType a = (RealDType a, Integral a, E.Integral (TensorElement a))

-- | Transcendental operations on host values and element expressions.
type FloatingDType a = (FractionalDType a, Floating a, Floating (TensorElement a))

-- | Ordered fractional arithmetic and rounding on host values and element expressions.
type RealFracDType a = (RealDType a, FractionalDType a, RealFrac a, E.RealFrac (TensorElement a))

-- | Floating-point arithmetic, comparisons, and classification.
type RealFloatDType a = (RealFracDType a, FloatingDType a, RealFloat a, E.RealFloat (TensorElement a))

-- | Equality comparisons on tensors of any shape.
type EqTensors t a = (forall sh. Shape sh => E.Eq (t sh a)) :: Constraint

-- | Ordered comparisons on tensors of any shape.
type OrdTensors t a = (forall sh. Shape sh => E.Ord (t sh a)) :: Constraint

-- | Numeric operations on tensors of any shape.
type NumTensors t a = (forall sh. Shape sh => Num (t sh a)) :: Constraint

-- | Fractional operations on tensors of any shape.
type FractionalTensors t a = (forall sh. Shape sh => Fractional (t sh a)) :: Constraint

-- | Integer division on tensors of any shape.
type IntegralTensors t a = (forall sh. Shape sh => E.Integral (t sh a)) :: Constraint

-- | Floating-point operations on tensors of any shape.
type FloatingTensors t a = (forall sh. Shape sh => Floating (t sh a)) :: Constraint

-- | Fractional decomposition and rounding on tensors of any shape.
type RealFracTensors t a = (forall sh. Shape sh => E.RealFrac (t sh a)) :: Constraint

-- | Floating-point arithmetic, decomposition, and classification on tensors of any shape.
type RealFloatTensors t a = (forall sh. Shape sh => E.RealFloat (t sh a)) :: Constraint

-- | Conditional selection on tensors of any shape; mask types are not constrained.
type ConditionalTensors t a = (forall sh. Shape sh => E.Conditional (t sh a)) :: Constraint

-- | Numeric element types and tensor arithmetic for every shape.
type NumTensorOps t a = (TensorOps t a, NumDType a, NumTensors t a)

-- | Fractional element types and tensor arithmetic for every shape.
type FractionalTensorOps t a = (NumTensorOps t a, FractionalDType a, FractionalTensors t a)

-- | Integer element types, tensor arithmetic, comparisons, and division for every shape.
type IntegralTensorOps t a = (NumTensorOps t a, IntegralDType a, OrdTensors t a, IntegralTensors t a)

-- | Floating-point element types and tensor arithmetic for every shape.
type FloatingTensorOps t a = (FractionalTensorOps t a, FloatingDType a, FloatingTensors t a)

-- | Ordered fractional element types, tensor arithmetic, and rounding for every shape.
type RealFracTensorOps t a = (FractionalTensorOps t a, RealFracDType a, RealFracTensors t a)

-- | Floating-point arithmetic, comparisons, classification, and conditional selection for every shape.
type RealFloatTensorOps t a =
    ( FloatingTensorOps t a
    , RealFracTensorOps t a
    , RealFloatDType a
    , OrdTensors t a
    , RealFloatTensors t a
    , ConditionalTensors t a
    )
