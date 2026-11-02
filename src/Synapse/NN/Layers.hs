-- | Layer parameters and ordinary functions applying them to tensors.
-- Parameters support execution, differentiation and weight serialization through their tensor representation.
module Synapse.NN.Layers
  ( module Synapse.NN.Layers.Dense,
    module Synapse.NN.Layers.Convolution,
  )
where

import Synapse.NN.Layers.Convolution
import Synapse.NN.Layers.Dense
