module FieldType where

import qualified Synapse.Autograd.Differentiable as D

bad :: D.Structure (Double, Float)
bad =
  D.record (,) (,)
    D.<+> D.field fst fst
    D.<+> D.field fst fst
