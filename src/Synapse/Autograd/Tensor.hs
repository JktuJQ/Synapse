{- | Differentiable tensors and tensor structures.
Use the operations from "Synapse.Tensor" with 'Tensor' inside a differentiation scope.
The same tensor function can then be evaluated normally or differentiated through "Synapse.Autograd.Execution".
'WithGrad' selects this representation for all tensor fields in a tuple or parameter record.

'Tensor' retains its input and operation history until a gradient boundary is introduced.
Use @constant@ to introduce fixed tensors and @stopGrad@ to detach a computed value.
Both functions are provided by "Synapse.Autograd.Execution".
-}
module Synapse.Autograd.Tensor
    ( -- * Tensors and structures
      Tensor
    , WithGrad
    ) where

import Synapse.Autograd.Tensor.Ops ()
import Synapse.Autograd.Tensor.Ops.Elementwise ()
import Synapse.Autograd.Tensor.Ops.Math ()
import Synapse.Autograd.Tensor.Representation (Tensor, WithGrad)
