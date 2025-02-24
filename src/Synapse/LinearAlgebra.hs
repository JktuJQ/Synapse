{- | Module that provides mathematical base for neural networks.

This module implements @Vec@ and @Mat@ datatypes and provides
several useful function to work with them.
-}


{-# LANGUAGE FunctionalDependencies #-}  -- @FunctionalDependencies@ are needed to define @ElementwiseScalarOps@ typeclass.
{-# LANGUAGE TypeFamilies           #-}  -- @TypeFamilies@ are needed to define @Indexable@ typeclass.


module Synapse.LinearAlgebra
    ( -- * Typeclasses
      Approx ((~==), (~/=), correct, roundTo)
    , Indexable (Index, unsafeIndex, index, safeIndex)
    , (!)
    , (!?)
    , ElementwiseScalarOps ((+.), (-.), (*.), (/.), (**.))

      -- * Constants
    , epsilon
    , closeToZero
    , closeToOne
    ) where


infix 4 ~==, ~/=
-- | @Approx@ class provides functions to work with potential floating point errors.
class Approx a where
    -- | Checks whether two objects are nearly equal.
    (~==) :: a -> a -> Bool
    (~==) a b = not $ (~/=) a b

    -- | Checks whether two objects are not nearly equal.
    (~/=) :: a -> a -> Bool
    (~/=) a b = not $ (~==) a b

    -- | Corrects distortions that may be caused by float operations.
    correct :: a -> Int -> a

    -- | Rounds to given amount of digits after floating point. Passing negative number shifts floating point to the left.
    roundTo :: a -> Int -> a

    {-# MINIMAL ((~==) | (~/=)), correct, roundTo #-}

-- | Constant that defines the difference between two fractionals that is considered small enough for approximate equality.
epsilon :: Fractional a => a
epsilon = 1e-5

-- | Constant that defines how small fractional part of number has to be to be considered close to zero.
closeToZero :: Fractional a => a
closeToZero = 0.0001

-- | Constant that defines how big fractional part of number has to be to be considered close to one.
closeToOne :: Fractional a => a
closeToOne = 0.9999

instance Approx Float where
    (~==) x y = let m = max (abs x) (abs y)
               in (m < epsilon) || ((abs (x - y) / m) < epsilon)

    correct x digits = if x == -0.0 then 0.0 else n / mul
      where
        mul = 10.0 ^ digits

        powered = x * mul
        n = if (closeToZero > fractionalPart) || (fractionalPart > closeToOne)
                then fromIntegral (round n :: Int)
                else n
          where
            (_, fractionalPart) = properFraction $ abs powered :: (Int, Float)

    roundTo x digits = let mul = 10.0 ^ digits
                       in fromIntegral (round (x * mul) :: Int) / mul

instance Approx Double where
    (~==) x y = let m = max (abs x) (abs y)
               in (m < epsilon) || ((abs (x - y) / m) < epsilon)

    correct x digits = if x == -0.0 then 0.0 else n / mul
      where
        mul = 10.0 ^ digits

        powered = x * mul
        n = if (closeToZero > fractionalPart) || (fractionalPart > closeToOne)
                then fromIntegral (round n :: Int)
                else n
          where
-- | @Indexable@ typeclass provides indexing interface for datatypes. 
class Indexable f where
    -- | Type of index for @Indexable@ collection.
    type Index f

    -- | Unsafe indexing.
    unsafeIndex :: f a -> Index f -> a

    -- | Indexing with bounds checking.
    index :: f a -> Index f -> a

    -- | Safe indexing.
    safeIndex :: f a -> Index f -> Maybe a


infixl 9 !
-- | Indexing with bounds checking (operator alias for @index@).
(!) :: Indexable f => f a -> Index f -> a
(!) = index

-- | Safe indexing (operator alias for @safeIndex@).
infixl 9 !?
(!?) :: Indexable f => f a -> Index f -> Maybe a
(!?) = safeIndex


infixl 6 +., -.
infixl 7 *., /.
infixr 8 **.
{- | @ElementwiseScalarOps@ class allows collections over numerical values easily work with scalars by using elementwise operations.

This class is a multiparameter class to permit instances on types that are not exactly collections, but rather wrappers of collections.
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
