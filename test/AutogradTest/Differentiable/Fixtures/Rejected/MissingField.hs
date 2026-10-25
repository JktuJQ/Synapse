module MissingField where

import qualified Synapse.Autograd.Differentiable as D

bad :: D.Structure Num (Double, Float)
bad =
  D.record (,) (,)
    D.<+> D.field fst fst
