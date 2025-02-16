{- | This module provides nessesary abstraction over layers of neural networks.

@AbstractLayer@ typeclass defines interface of all layers of neural network model.
Its implementation is probably the most low-leveled abstraction of the @Synapse@ library.
Notes on how to correctly implement that typeclass are in the docs for it.

@Layer@ is the existential datatype that wraps any @AbstractLayer@ instance.
That is the building block of any neural network.
-}


{-# LANGUAGE DefaultSignatures         #-}  -- @DefaultSignatures@ are needed to provide default implementation in @AbstractLayer@ typeclass.
{-# LANGUAGE ExistentialQuantification #-}  -- @ExistentialQuantification@ is needed to define @Layer@ datatype.
{-# LANGUAGE FlexibleInstances         #-}  -- @FlexibleInstances@ are needed to define and implement @AbstractLayer@ typeclass.
{-# LANGUAGE FunctionalDependencies    #-}  -- @FunctionalDependencies@ are needed to define and implement @AbstractLayer@ typeclass.


module Synapse.ML.Layers.Layer
    ( -- * @AbstractLayer@ typeclass

      AbstractLayer (symbolicForward, forward, getParameters, updateParameters)
      
      -- * @Layer@ existential datatype

    , Layer (Layer)
    ) where


import Synapse.LinearAlgebra.Mat (Mat)

import Synapse.Autograd (Symbol(unSymbol), Symbolic, constSymbol)


{- | @AbstractLayer@ typeclass defines basic interface of all layers of neural network model.

Every layer should be able to pass (@forward@) @Mat@ through itself to produce new @Mat@ - make prediction based on its parameters.
There is also @symbolicForward@ counterpart, which allows for gradients to be calculated after predictions,
which in turn makes training possible.

@getParameters@ function is needed to obtain information about number and and form of parameters on a layer.
That allows optimiser to work with any layer that provides such information.
@updateParameters@ function is given a list of changed parameters
(length of that list, the order and the form of parameters is EXACTLY the same as those from @getParameters@)
and layer should change itself accordingly.

Note: this typeclass correct implementation is very important for work of the neural network and training,
read the docs thoroughly to ensure that all the invariants are met.
-}
class Num a => AbstractLayer l a | l -> a where
    {- | Passes symbolic matrix through the layer to produce new symbolic matrix, while retaining gradients graph.

    Given @String@ is a prefix for name of symbolic parameters that are used in calculation.
    Every used parameter should have unique name to be recognised by the autograd - 
    it must start with given prefix and end with the numerical index of said parameter.
    For example 3rd layer with 2 parameters (weights and bias) should
    name its weights symbol "l3w1" and name its bias symbol "l3w2" ("l3w" prefix will be supplied).
    It is also important so that the order of the parameters stays consistent even for @getParameters@ function
    (that will allow choosing correct gradients automatically in the training).
    -}
    symbolicForward :: Symbolic a => String -> l -> Symbol (Mat a) -> Symbol (Mat a)

    -- | Passes matrix through the layer to produce new matrix.
    forward :: l -> Mat a -> Mat a
    default forward :: Symbolic a => l -> Mat a -> Mat a
    forward l mat = unSymbol $ symbolicForward "" l (constSymbol mat)

    -- | Returns a list of all parameters of the layer (those must be of the exact same order as they are named (check @symbolicForward@ docs)).
    getParameters :: l -> [Mat a]

    -- | Updates parameters of the layer based on supplied list (length of that list, the order and the form of parameters is EXACTLY the same as those from @getParameters@)
    updateParameters :: l -> [Mat a] -> l


-- | @Layer@ existential datatype wraps anything that implements @AbstractLayer@.
data Layer a = forall l. AbstractLayer l a => Layer l
