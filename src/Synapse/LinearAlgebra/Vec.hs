{- | Implementation of mathematical vector.

@Vec@ is only a newtype wrapper around @Vector@, which
implements several mathematical operations on itself.

@Vec@ offers meaningful abstraction and easy interface
(you can unwrap it to perform more complex tasks).
-}


{- @TypeFamilies@ are needed to instantiate @Container@, @Indexable@, @ElementwiseScalarOps@, @SingletonOps@, @VecOps@, @MatOps@ typeclasses.
-}

{-# LANGUAGE TypeFamilies #-}


module Synapse.LinearAlgebra.Vec
    ( -- * @Vec@ datatype and simple getters.

      Vec (Vec, unVec)

    , size

      -- * Constructors

    , empty
    , singleton
    , fromList
    , generate
    , replicate

      -- * Concatenation and splitting

    , cons
    , snoc
    , (++)
    , concat

    , splitAt

      -- * Combining

    , map
    , imap
    , for
    , zipWith
    , zip

    -- * Mathematics

    , zeroes
    , ones

    , squaredMagnitude
    , magnitude
    , clampMagnitude
    , normalized

    , linearCombination
    , dot

    , angleBetween
    , lerp
    ) where


import Synapse.LinearAlgebra (Container(..), Indexable(..), ElementwiseScalarOps(..), SingletonOps(..), VecOps(..))

import Prelude hiding ((++), concat, splitAt, map, replicate, zip, zipWith)
import Data.Foldable (Foldable(..))
import Data.Ord (clamp)

import qualified Data.Vector as V


-- | Mathematical vector (collection of elements).
newtype Vec a = Vec
    { unVec :: V.Vector a  -- ^ Internal representation.
    } deriving (Eq, Read)

-- | Size of a vector - number of elements.
size :: Vec a -> Int
size = V.length . unVec


-- Typeclasses

instance Show a => Show (Vec a) where
    show (Vec x) = show x


instance Container (Vec a) where
    type DType (Vec a) = a


instance Indexable (Vec a) where
    type Index (Vec a) = Int

    unsafeIndex (Vec x) = V.unsafeIndex x
    (!) (Vec x) = (V.!) x
    (!?) (Vec x) = (V.!?) x


instance Num a => Num (Vec a) where
    (+) = zipWith (+)
    (-) = zipWith (-)
    negate = fmap (0 -)
    (*) = zipWith (*)
    abs = fmap abs
    signum = fmap signum
    fromInteger = singleton . fromInteger

instance Fractional a => Fractional (Vec a) where
    (/) = zipWith (/)
    recip = fmap (1/)
    fromRational = singleton . fromRational

instance Floating a => Floating (Vec a) where
    pi = singleton pi
    (**) = zipWith (**)
    sqrt = fmap sqrt
    exp = fmap exp
    log = fmap log
    sin = fmap sin
    cos = fmap cos
    asin = fmap asin
    acos = fmap acos
    atan = fmap atan
    sinh = fmap sinh
    cosh = fmap cosh
    asinh = fmap asinh
    acosh = fmap acosh
    atanh = fmap atanh

instance ElementwiseScalarOps (Vec a) where
    (+.) x n = fmap (+ n) x
    (-.) x n = fmap (subtract n) x
    (*.) x n = fmap (* n) x
    (/.) x n = fmap (/ n) x
    (**.) x n = fmap (** n) x

    elementsMin x n = fmap (min n) x
    elementsMax x n = fmap (max n) x

instance SingletonOps (Vec a) where
    singleton = pure
    unSingleton v
        | size v /= 1 = error "Vector is not a singleton"
        | otherwise   = unsafeIndex v 0

    elementsSum = singleton . V.sum . unVec
    elementsProduct = singleton . V.product . unVec
    mean x = elementsSum x /. fromIntegral (size x)
    norm x = sqrt $ elementsSum $ x * x

instance Functor Vec where
    fmap f = Vec . V.map f . unVec
    (<$) = fmap . const

instance Applicative Vec where
    pure = Vec . V.singleton
    (<*>) = zipWith (\f x -> f x)

instance Foldable Vec where
    foldr f x = V.foldr f x . unVec
    foldl f x = V.foldl f x . unVec

    foldr' f x = V.foldr' f x . unVec
    foldl' f x = V.foldl' f x . unVec

    foldr1 f = V.foldr1 f . unVec
    foldl1 f = V.foldl1 f . unVec

    toList = V.toList . unVec

    null x = size x == 0

    length = size

instance Traversable Vec where
    traverse f (Vec x) = Vec <$> traverse f x


-- Constructors

-- | Creates empty @Vec@.
empty :: Vec a
empty = Vec V.empty

-- | Creates @Vec@ from list.
fromList :: [a] -> Vec a
fromList = Vec . V.fromList

-- | Creates @Vec@ of given length using generating function.
generate :: Int -> (Int -> a) -> Vec a
generate n = Vec . V.generate n

-- | Creates @Vec@ of given length filled with given element.
replicate :: Int -> a -> Vec a
replicate n = generate n . const


-- Concatenation and splitting

-- | Prepend @Vec@ with given element.
cons :: a -> Vec a -> Vec a
cons x = Vec . V.cons x . unVec

-- | Append @Vec@ with given element.
snoc :: Vec a -> a -> Vec a
snoc (Vec v) x = Vec $ V.snoc v x

-- | Concatenate two @Vec@s.
infixr 5 ++
(++) :: Vec a -> Vec a -> Vec a
(++) (Vec x) (Vec y) = Vec $ x V.++ y

-- | Concatenate all @Vec@s.
concat :: [Vec a] -> Vec a
concat = foldr1 (++)

-- | Splits @Vec@ into two @Vec@s at a given index.
splitAt :: Int -> Vec a -> (Vec a, Vec a)
splitAt i (Vec v) = let (v1, v2) = V.splitAt i v
                  in (Vec v1, Vec v2)


-- Combining

-- | Map a function over a @Vec@.
map :: (a -> b) -> Vec a -> Vec b
map = fmap

-- | Apply a function to every element of a @Vec@ and its index.
imap :: (Int -> a -> b) -> Vec a -> Vec b
imap f = Vec . V.imap f . unVec

-- | @map@ with its arguments flipped.
for :: Vec a -> (a -> b) -> Vec b
for = flip fmap

-- | Zips two @Vec@s with the given function.
zipWith :: (a -> b -> c) -> Vec a -> Vec b -> Vec c
zipWith f (Vec a) (Vec b) = Vec $ V.zipWith f a b

-- | Zips two @Vec@s.
zip :: Vec a -> Vec b -> Vec (a, b)
zip = zipWith (,)


-- Functions that work on mathematical vector (type constraint refers to a number)

-- | Creates @Vec@ of given length filled with zeroes.
zeroes :: Num a => Int -> Vec a
zeroes = flip generate (const 0)

-- | Creates @Vec@ of given length filled with ones.
ones :: Num a => Int -> Vec a
ones = flip generate (const 1)


-- | Squared magnitude of a @Vec@.
squaredMagnitude :: Num a => Vec a -> a
squaredMagnitude x = sum (fmap (^ (2 :: Int)) x)

-- | Magnitude of a @Vec@.
magnitude :: Floating a => Vec a -> a
magnitude = sqrt . squaredMagnitude

-- | Clamps @Vec@ magnitude.
clampMagnitude :: (Floating a, Ord a) => a -> Vec a -> Vec a
clampMagnitude m x = x *. (min (magnitude x) m / magnitude x)

-- | Normalizes @Vec@ by dividing each component by @Vec@ magnitude.
normalized :: Floating a => Vec a -> Vec a
normalized x = x /. magnitude x


-- | Computes linear combination of @Vec@s. Returns empty @Vec@ if empty list was passed to this function.
linearCombination :: Num a => [(a, Vec a)] -> Vec a
linearCombination [] = empty
linearCombination (x:xs) = foldl' (\acc (a, v) -> acc + v *. a) (snd x *. fst x) xs

instance Num a => VecOps (Vec a) where
    dot a b = elementsSum $ a * b

-- | Calculates an angle between two @Vec@s.
angleBetween :: Floating a => Vec a -> Vec a -> a
angleBetween a b = acos $ unSingleton (a `dot` b) / (magnitude a * magnitude b)

-- | Linearly interpolates between two @Vec@s. Given parameter will be clamped between [0.0, 1.0].
lerp :: (Floating a, Ord a) => a -> Vec a -> Vec a -> Vec a
lerp k a b = b - (b - a) *. clamp (0.0, 1.0) k
