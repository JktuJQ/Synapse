{- | Module that provides mathematical base for neural networks.

This module implements @Vec@ and @Mat@ datatypes and provides
several useful function to work with them.

Most of typeclasses of this module are multiparameter typeclasses.
That is to permit instances on types that are not exactly collections, but rather wrappers of collections,
and it allows imposing additional constraints on inner type.
The best example is @Symbol@ from @Synapse.Autograd@.
-}


{-# LANGUAGE FunctionalDependencies #-}  -- @FunctionalDependencies@ are needed to define @ElementwiseScalarOps@, @SingletonOps@, @VecOps@, @MatOps@ typeclasses.
{-# LANGUAGE TypeFamilies           #-}  -- @TypeFamilies@ are needed to define @Indexable@ typeclass.


module Synapse.LinearAlgebra
    ( -- * @Indexable@ typeclass

     Indexable (Index, unsafeIndex, (!), (!?))

      -- * Scalar operations
    , ElementwiseScalarOps ((+.), (-.), (*.), (/.), (**.), elementsMin, elementsMax)
    , SingletonOps (singleton, unSingleton, elementsSum, elementsProduct, mean, norm)

      -- * Collection operations
    , VecOps (dot)
    , MatOps (transpose, matMul)
    ) where


infixl 9 !, !?
-- | @Indexable@ typeclass provides indexing interface for datatypes. 
class Indexable f where
    -- | Type of index for @Indexable@ collection.
    type Index f

    -- | Unsafe indexing.
    unsafeIndex :: f a -> Index f -> a

    -- | Indexing with bounds checking.
    (!) :: f a -> Index f -> a

    -- | Safe indexing.
    (!?) :: f a -> Index f -> Maybe a


infixl 6 +., -.
infixl 7 *., /.
infixr 8 **.
{- | @ElementwiseScalarOps@ typeclass allows collections over numerical values easily work with scalars by using elementwise operations.

This typeclass is a multiparameter typeclass to permit instances on types that are not exactly collections, but rather wrappers of collections.
The best example is @Symbol@ from @Synapse.Autograd@.
-}
class ElementwiseScalarOps f a | f -> a where
    -- | Adds given value to every element of the collection.
    (+.) :: Num a => f -> a -> f
    -- | Subtracts given value from every element of the functor.
    (-.) :: Num a => f -> a -> f
    -- | Multiplies every element of the functor by given value.
    (*.) :: Num a => f -> a -> f
    -- | Divides every element of the functor by given value.
    (/.) :: Fractional a => f -> a -> f
    -- | Exponentiates every element of the functor by given value.
    (**.) :: Floating a => f -> a -> f

    -- | Applies @min@ operation with given value to every element.
    elementsMin :: Ord a => f -> a -> f
    -- | Applies @max@ operation with given value to every element.
    elementsMax :: Ord a => f -> a -> f

{- | @Singleton@ typeclass provides operations that relate to singleton collections (scalars that are wrapped in said collection).

All functions of that typeclass must return singletons (scalars that are wrapped in collection).

This typeclass is a multiparameter typeclass to permit instances on types that are not exactly collections, but rather wrappers of collections.
The best example is @Symbol@ from @Synapse.Autograd@.
-}
class SingletonOps f a | f -> a where
    -- | Initializes singleton collection.
    singleton :: a -> f
    -- | Unwraps singleton collection.
    unSingleton :: f -> a

    -- | Sums all elements of collection.
    elementsSum :: Num a => f -> f
    -- | Multiplies all elements of collection (@Fractional@ constraint is needed for efficient gradient calculation).
    elementsProduct :: Fractional a => f -> f
    
    -- | Calculates the mean of all elements of collection.
    mean :: Fractional a => f -> f

    -- | Calculates the Frobenius norm of all elements of collection.
    norm :: Floating a => f -> f


{- | @VecOps@ typeclass provides vector-specific operations.

This typeclass is a multiparameter typeclass to permit instances on types that are not exactly collections, but rather wrappers of collections.
The best example is @Symbol@ from @Synapse.Autograd@.
-}
class VecOps f a | f -> a where
    -- | Calculates dot product of two vectors.
    dot :: Num a => f -> f -> f

{- | @MatOps@ typeclass provides matrix-specific operations.

This typeclass is a multiparameter typeclass to permit instances on types that are not exactly collections, but rather wrappers of collections.
The best example is @Symbol@ from @Synapse.Autograd@. 
-}
class MatOps f a | f -> a where
    -- | Transposes matrix.
    transpose :: f -> f

    -- | Mutiplies two matrices.
    matMul :: Num a => f -> f -> f
