-- | Layer parameters and ordinary functions applying them to tensors.
-- Parameters support execution, differentiation and weight serialization through their tensor representation.
module Synapse.NN.Layers
  ( module Synapse.NN.Layers.Dense,
    module Synapse.NN.Layers.Convolution,
    module Synapse.NN.Layers.Normalization,
    module Synapse.NN.Layers.Dropout,
    module Synapse.NN.Layers.Embedding,
  )
where

import Synapse.NN.Layers.Convolution
import Synapse.NN.Layers.Dense
import Synapse.NN.Layers.Dropout
import Synapse.NN.Layers.Embedding
import Synapse.NN.Layers.Normalization
