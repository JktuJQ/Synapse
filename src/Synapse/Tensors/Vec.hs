{- | Implementation of mathematical vector.

@Vec@ is only a newtype wrapper around @Vector@, which
implements several mathematical operations on itself.

@Vec@ offers meaningful abstraction and easy interface
(you can unwrap it to perform more complex tasks).
-}


-- This language pragma is needed to support @Indexable@ typeclass.
{-# LANGUAGE TypeFamilies #-}


module Synapse.Tensors.Vec
    ( -- * @Vec@ datatype and simple getters.

      Vec (Vec, unVec)

    , size

      -- * Constructors

    , empty
    , singleton
    , fromList
    , generate
    , replicate

      -- * Concatenation

    , cons
    , snoc
    , (++)
    , concat
    
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


import Synapse.Tensors (Approx(..), Indexable(..))

import Prelude hiding ((++), concat, map, replicate, zip, zipWith)
import Data.Foldable (Foldable(..))
import Data.Ord (clamp)

import qualified Data.Vector as V


-- | Mathematical vector (collection of elements).
newtype Vec a = Vec
    { unVec :: V.Vector a  -- ^ Internal representation
    } deriving (Eq, Read, Show)

-- | Size of a vector - number of elements.
size :: Vec a -> Int
size = V.length . unVec


-- Typeclasses

instance Indexable Vec where
    type Index Vec = Int

    unsafeIndex (Vec x) = V.unsafeIndex x

    index (Vec x) = (V.!) x

    safeIndex (Vec x) = (V.!?) x


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

instance Approx a => Approx (Vec a) where
    (~==) x y = and $ zipWith (~==) x y

    correct x digits = fmap (`correct` digits) x
    roundTo x digits = fmap (`roundTo` digits) x


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

-- | Creates @Vec@ that contains only one element.
singleton :: a -> Vec a
singleton = pure

-- | Creates @Vec@ from list.
fromList :: [a] -> Vec a
fromList = Vec . V.fromList

-- | Creates @Vec@ of given length using generating function.
generate :: Int -> (Int -> a) -> Vec a
generate n = Vec . V.generate n

-- | Creates @Vec@ of given length filled with given element.
replicate :: Int -> a -> Vec a
replicate n = generate n . const


-- Concatenation

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
squaredMagnitude = sum . map (^ (2 :: Int))

-- | Magnitude of a @Vec@.
magnitude :: Floating a => Vec a -> a
magnitude = sqrt . squaredMagnitude

-- | Clamps @Vec@ magnitude.
clampMagnitude :: (Floating a, Ord a) => a -> Vec a -> Vec a
clampMagnitude m x = map (* (min (magnitude x) m / magnitude x)) x

-- | Normalizes @Vec@ by dividing each component by @Vec@ magnitude.
normalized :: Floating a => Vec a -> Vec a
normalized x = map (/ magnitude x) x


-- | Computes linear combination of @Vec@s. Returns empty @Vec@ if empty list was passed to this function.
linearCombination :: Num a => [(a, Vec a)] -> Vec a
linearCombination [] = empty
linearCombination (x:xs) = foldl' (\acc (a, v) -> acc + fmap (* a) v) (fmap (* fst x) (snd x)) xs

-- | Calculates dot product of two @Vec@s.
dot :: Num a => Vec a -> Vec a -> a
dot a b = sum $ zipWith (*) a b


-- | Calculates an angle between two @Vec@s.
angleBetween :: Floating a => Vec a -> Vec a -> a
angleBetween a b = acos ((a `dot` b) / (magnitude a * magnitude b))

-- | Linearly interpolates between two @Vec@s. Given parameter will be clamped between [0.0, 1.0].
lerp :: (Floating a, Ord a) => a -> Vec a -> Vec a -> Vec a
lerp k a b = b - fmap (* clamp (0.0, 1.0) k) (b - a)
