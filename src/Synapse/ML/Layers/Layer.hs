{- | This module provides nessesary abstraction over layers of neural networks.

@AbstractLayer@ typeclass defines interface of all layers of neural network model.
Its implementation is probably the most low-leveled abstraction of the @Synapse@ library.
Notes on how to correctly implement that typeclass are in the docs for it.

@Layer@ is the existential datatype that wraps any @AbstractLayer@ instance.
That is the building block of any neural network.
-}


{-# LANGUAGE ExistentialQuantification #-}  -- @ExistentialQuantification@ is needed to define @Layer@ datatype.


module Synapse.ML.Layers.Layer
    ( -- * @AbstractLayer@ typeclass

      AbstractLayer (inputSize, outputSize, getParameters, updateParameters, symbolicForward)

    , forward
      
      -- * @Layer@ existential datatype

    , Layer (Layer)

      -- * @LayerConfiguration@ type alias
    , LayerConfiguration
    ) where


import Synapse.LinearAlgebra.Mat (Mat)

import Synapse.Autograd (Symbolic, Symbol(unSymbol), constSymbol)


{- | @AbstractLayer@ typeclass defines basic interface of all layers of neural network model.

Every layer should be able to pass @Mat@ through itself to produce new @Mat@ (make prediction based on its parameters)
using @symbolicForward@ function, which allows for gradients to be calculated after predictions,
which in turn makes training possible.

@getParameters@ function is needed to obtain information about number and and form of parameters on a layer.
That allows optimiser to work with any layer that provides such information.
@updateParameters@ function is given a list of changed parameters
(length of that list, the order and the form of parameters is EXACTLY the same as those from @getParameters@)
and layer should change itself accordingly.

Note: any model is an @AbstractLayer@ instance too, so don't be confused by the docs on some methods.

Important: this typeclass correct implementation is very important for work of the neural network and training,
read the docs thoroughly to ensure that all the invariants are met.
-}
class AbstractLayer l where
    -- | Returns the size of the input for @forward@ and @symbolicForward@ functions that is supported. @Nothing@ means size independence (activation functions are the example).
    inputSize :: l a -> Maybe Int
    -- | Returns the size of the output of @forward@ and @symbolicForward@. @Nothing@ means size independence (activation functions are the example).
    outputSize :: l a -> Maybe Int

    -- | Returns a list of all parameters (those must be of the exact same order as they are named (check @symbolicForward@ docs)).
    getParameters :: l a -> [Mat a]

    -- | Updates parameters based on supplied list (length of that list, the order and the form of parameters is EXACTLY the same as those from @getParameters@)
    updateParameters :: l a -> [Mat a] -> l a

    {- | Passes symbolic matrix through to produce new symbolic matrix, while retaining gradients graph.

    Given @String@ is a prefix for name of symbolic parameters that are used in calculation.
    Every used parameter should have unique name to be recognised by the autograd - 
    it must start with given prefix and end with the numerical index of said parameter.
    For example 3rd layer with 2 parameters (weights and bias) should
    name its weights symbol "l3w1" and name its bias symbol "l3w2" ("l3w" prefix will be supplied).
    It is also important so that the order of the parameters stays consistent even for @getParameters@ function
    (that will allow choosing correct gradients automatically in the training).
    -}
    symbolicForward :: (Symbolic a, Floating a, Ord a) => String -> l a -> Symbol (Mat a) -> Symbol (Mat a)


-- | Passes matrix through to produce new matrix.
forward :: (AbstractLayer l, Symbolic a, Floating a, Ord a) => l a -> Mat a -> Mat a
forward layer input = unSymbol $ symbolicForward "" layer (constSymbol input)


-- | @Layer@ existential datatype wraps anything that implements @AbstractLayer@.
data Layer a = forall l. AbstractLayer l => Layer (l a)

instance AbstractLayer Layer where
    inputSize (Layer l) = inputSize l
    outputSize (Layer l) = outputSize l

    getParameters (Layer l) = getParameters l
    updateParameters (Layer l) = Layer . updateParameters l

    symbolicForward prefix (Layer l) = symbolicForward prefix l


-- | @LayerConfiguration@ type alias represents functions that are able to build layers.
type LayerConfiguration l
    =  Int  -- ^ Output size of previous layer.
    -> l    -- ^ Resulting layer.
