{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module MissingConstraint where

import qualified Synapse.Autograd as Base
import Synapse.Autograd.Differentiable

data Parameters = Parameters Double Float

data Variables s = Variables (Base.Var s Double) (Base.Var s Float)

first :: Parameters -> Double
first (Parameters x _) = x

second :: Parameters -> Float
second (Parameters _ y) = y

firstVar :: Variables s -> Base.Var s Double
firstVar (Variables x _) = x

secondVar :: Variables s -> Base.Var s Float
secondVar (Variables _ y) = y

instance Differentiable Parameters where
  type WithGrad s Parameters = Variables s
  type ParametersConstraints c Parameters = FieldConstraints c '[Double]
  structureWith = record Parameters Variables <+> field first firstVar <+> field second secondVar
