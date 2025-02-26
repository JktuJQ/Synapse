{- | Implementation of matrix.

Matrices are a backbone of any machine learning library,
since most of the operations are implemented by the matrices
combinations (matrix multiplication, elementwise operations).

@Mat@ datatype provides interface for all of those operations.
-}


{-# LANGUAGE FlexibleInstances     #-}  -- @FlexibleInstances@ are needed to implement @ElementwiseScalarOps@ typeclass.
{-# LANGUAGE MultiParamTypeClasses #-}  -- @MultiParamTypeClasses@ are needed to implement @Indexable@ and @ElementwiseScalarOps@ typeclasses.
{-# LANGUAGE TypeFamilies          #-}  -- @TypeFamilies@ are needed to implement @Indexable@ typeclass.


module Synapse.LinearAlgebra.Mat
    ( --  * @Mat@ datatype and simple getters.

      Mat (nRows, nCols)

    , nElements
    , size

    , unSingleton

    , isTransposed
    , isSubmatrix

      -- * Utility

    , force
    , toLists

      -- * Constructors

    , empty
    , singleton
    , fromList
    , fromLists
    , generate
    , replicate

      -- * Vec operations

    , rowVec
    , colVec
    , fromVec

    , indexRow
    , indexCol
    , safeIndexRow
    , safeIndexCol

    , diagonal
    , flatten

     -- * Combining

    , map
    , mapRow
    , mapCol
    , for
    , imap
    , elementwise

      -- * Operations with matrices

    , setSize
    , extend
    , shrink

    , swapRows
    , swapCols
    , transpose

      -- * Submatrices

    , minor
    , submatrix
    , split
    , join
    , (<|>)
    , (<->)

      -- * Mathematics

    , zeroes
    , ones
    , identity

    , adamarMul
    , matMul

    , det
    , rref
    , inverse
    , orthogonalized
    ) where


import Synapse.LinearAlgebra (Indexable(..), (!), ElementwiseScalarOps(..), ToScalarOps(..), MatOps(..))

import Synapse.LinearAlgebra.Vec (Vec(Vec))
import qualified Synapse.LinearAlgebra.Vec as SV

import Prelude hiding (map, replicate, zip)
import Data.Foldable (Foldable(..))
import Data.List (find)
import Data.Tuple (swap)

import qualified Data.Vector as V


{- | Mathematical matrix (collection of elements).

This implementation focuses on sharing parts of matrices and clever indexation
to reduce overhead of several essential operations.
Those include splitting matrices into submatrices, transposing - their asymptotical complexity becomes O(1).
However there are few downsides:
the first is that severely splitted matrix is hard to garbage collect and is not cache-friendly
and the second is that mass traversal operations on those sparse matrices might not fuse and combine well.
@force@ and @forced*@ functions address those issues, but most of the time those problems are not
significant enough and you are just better using convenient functions instead of workarounds.
-}
data Mat a = Mat
    { nRows        :: {-# UNPACK #-} !Int         -- ^ Number of rows. 
    , nCols        :: {-# UNPACK #-} !Int         -- ^ Number of columns.
    , rowStride    :: {-# UNPACK #-} !Int         -- ^ How much increasing row index affects true indexing.
    , colStride    :: {-# UNPACK #-} !Int         -- ^ How much increasing column index affects true indexing.
    , rowOffset    :: {-# UNPACK #-} !Int         -- ^ Row offset (from which row index does the matrix actually start).
    , colOffset    :: {-# UNPACK #-} !Int         -- ^ Column offset (from which column index does the matrix actually start).
    , storage      ::                 V.Vector a  -- ^ Internal storage (elements are stored in a vector using row-major ordering).
    }

-- | Number of elements in a matrix.
nElements :: Mat a -> Int
nElements mat = nRows mat * nCols mat

-- | Size of matrix.
size :: Mat a -> (Int, Int)
size mat = (nRows mat, nCols mat)


-- | Extracts scalar element if @Mat@ is a singleton.
unSingleton :: Mat a -> a
unSingleton mat
    | nElements mat /= 1 = error "Vector is not a singleton"
    | otherwise   = unsafeIndex mat (0, 0)


-- | Whether the matrix is transposed. If the matrix consists of only one element, it is considered never transposed.
isTransposed :: Mat a -> Bool
isTransposed mat = colStride mat /= 1 && rowStride mat == 1

-- | Returns whether the matrix is a submatrix from another matrix.
isSubmatrix :: Mat a -> Bool
isSubmatrix mat = any (0 /=) [rowOffset mat, colOffset mat]


-- | Converts two dimensional matrix index to one dimensional vector index.
indexMatToVec :: Mat a -> (Int, Int) -> Int
indexMatToVec (Mat _ _ rk ck r0 c0 _) (r, c) = (r0 + r) * rk + (c0 + c) * ck

-- | Converts one dimensional vector index to two dimensional matrix index.
indexVecToMat :: Mat a -> Int -> (Int, Int)
indexVecToMat mat@(Mat _ _ rk ck r0 c0 _) i = let t = isTransposed mat
                                                  (r', c') = (if t then swap else id) $ quotRem i (if t then ck else rk)
                                              in (r' - r0, c' - c0)



-- | Copies matrix data dropping any extra memory that may be held if given matrix is a submatrix.
force :: Mat a -> Mat a
force x = Mat (nRows x) (nCols x) (nCols x) 1 0 0 (V.fromList [unsafeIndex x (r, c) | r <- [0 .. nRows x - 1], c <- [0 .. nCols x - 1]])

-- | Converts matrix to list of lists.
toLists :: Mat a -> [[a]]
toLists x = [[unsafeIndex x (r, c) | c <- [0 .. nCols x - 1] ] | r <- [0 .. nRows x - 1]]


-- Typeclasses

instance Show a => Show (Mat a) where
    show mat = "(" ++ show (size mat) ++ "): " ++ show (toLists mat)


instance Indexable Mat where
    type Index Mat = (Int, Int)

    unsafeIndex x (r, c) = V.unsafeIndex (storage x) (indexMatToVec x (r, c))
    (!) x (r, c)
        | r < 0 || r >= nRows x || c < 0 || c >= nCols x = error $ "Index " ++ show (r, c) ++ " is out of bounds for matrix with size " ++ show (size x)
        | otherwise                                      = unsafeIndex x (r, c)
    (!?) x (r, c)
        | r < 0 || r >= nRows x || c < 0 || c >= nCols x = Nothing
        | otherwise                                      = Just $ unsafeIndex x (r, c)


instance Num a => Num (Mat a) where
    (+) = elementwise (+)
    (-) = elementwise (-)
    negate = fmap (0 -)
    (*) = adamarMul
    abs = fmap abs
    signum = fmap signum
    fromInteger = singleton . fromInteger

instance Fractional a => Fractional (Mat a) where
    (/) = elementwise (/)
    recip = fmap (1/)
    fromRational = singleton . fromRational

instance Floating a => Floating (Mat a) where
    pi = singleton pi
    (**) = elementwise (**)
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


instance ElementwiseScalarOps (Mat a) a where
    (+.) x n = fmap (+ n) x
    (-.) x n = fmap (subtract n) x
    (*.) x n = fmap (* n) x
    (/.) x n = fmap (/ n) x
    (**.) x n = fmap (** n) x

    elementsMin x n = fmap (min n) x
    elementsMax x n = fmap (max n) x

instance ToScalarOps (Mat a) a where
    elementsSum mat@(Mat rows cols _ _ _ _ _) = singleton $ sum [unsafeIndex mat (r, c) | r <- [0 .. rows - 1], c <- [0 .. cols - 1]]
    elementsProduct mat@(Mat rows cols _ _ _ _ _) = singleton $ product [unsafeIndex mat (r, c) | r <- [0 .. rows - 1], c <- [0 .. cols - 1]]

    mean x = elementsSum x /. fromIntegral (nElements x)

    norm x = sqrt $ elementsSum $ x * x


instance Eq a => Eq (Mat a) where
    (==) a@(Mat rows1 cols1 _ _ _ _ _) b@(Mat rows2 cols2 _ _ _ _ _)
        | rows1 /= rows2 || cols1 /= cols2 = False
        | otherwise                        = and [unsafeIndex a (r, c) == unsafeIndex b (r, c) | r <- [0 .. rows1 - 1], c <- [0 .. cols1 - 1]]


instance Functor Mat where
    fmap f (Mat rows cols rk ck r0 c0 x) = Mat rows cols rk ck r0 c0 (fmap f x)
    (<$) = fmap . const

instance Applicative Mat where
    pure x = Mat 1 1 1 1 0 0 (V.singleton x)
    (<*>) = elementwise (\f x -> f x)

instance Foldable Mat where
    foldr f x = V.foldr f x . storage
    foldl f x = V.foldl f x . storage

    foldr' f x = V.foldr' f x . storage
    foldl' f x = V.foldl' f x . storage

    foldr1 f = V.foldr1 f . storage
    foldl1 f = V.foldl1 f . storage

    toList = V.toList . storage

    null x = nElements x == 0

    length = nElements

instance Traversable Mat where
    sequenceA mat@(Mat rows cols _ _ _ _ _) = fmap (Mat rows cols cols 1 0 0) . sequenceA . storage $ force mat


-- Constructors

-- | Creates empty @Mat@.
empty :: Mat a
empty = Mat 0 0 0 0 0 0 V.empty

-- | Creates @Mat@ that consists of only one element.
singleton :: a -> Mat a
singleton x = Mat 1 1 1 1 0 0 (V.singleton x)

-- | Creates @Mat@ from list (will throw an error, if elements of that list do not form a matrix of given size).
fromList :: (Int, Int) -> [a] -> Mat a
fromList (rows, cols) xs
    | V.length m /= rows * cols = error "Given dimensions do not match with list length"
    | otherwise                 = Mat rows cols cols 1 0 0 m
  where
    m = V.fromList xs

-- | Creates @Mat@ from list of lists (alias for @fromLists (rows, cols) (concat xs)@).
fromLists :: (Int, Int) -> [[a]] -> Mat a
fromLists (rows, cols) xs = fromList (rows, cols) (concat xs)

-- | Creates @Mat@ of given size using generating function.
generate :: (Int, Int) -> ((Int, Int) -> a) -> Mat a
generate (rows, cols) f = Mat rows cols cols 1 0 0 (V.generate (rows * cols) (f . flip quotRem cols))

-- | Creates @Mat@ of given size filled with given element.
replicate :: (Int, Int) -> a -> Mat a
replicate (rows, cols) x = Mat rows cols cols 1 0 0 (V.replicate (rows * cols) x)


-- Vec operations

-- | Converts @Vec@ to a one row @Mat@.
rowVec :: Vec a -> Mat a
rowVec (Vec x) = Mat 1 (V.length x) (V.length x) 1 0 0 x

-- | Converts @Vec@ to a one column @Mat@.
colVec :: Vec a -> Mat a
colVec (Vec x) = Mat (V.length x) 1 1 1 0 0 x

-- | Initializes @Mat@ from given @Vec@.
fromVec :: (Int, Int) -> Vec a -> Mat a
fromVec (rows, cols) (Vec x)
    | V.length x /= rows * cols = error "Given dimensions do not match with vector length"
    | otherwise                 = Mat rows cols cols 1 0 0 x


-- | Extracts row from @Mat@. If row is not present, an error is thrown.
indexRow :: Mat a -> Int -> Vec a
indexRow mat@(Mat rows cols _ _ _ _ x) r
    | r < 0 || r >= rows = error "Given row is not present in the matrix"
    | isTransposed mat   = Vec $ V.fromList [unsafeIndex mat (r, c) | c <- [0 .. cols - 1]]
    | otherwise          = Vec $ V.slice (indexMatToVec mat (r, 0)) cols x

-- | Extracts column from @Mat@. If column is not present, an error is thrown.
indexCol :: Mat a -> Int -> Vec a
indexCol mat@(Mat rows cols _ _ _ _ x) c
    | c < 0 || c >= cols = error "Given column is not present in the matrix"
    | isTransposed mat   = Vec $ V.slice (indexMatToVec mat (0, c)) rows x
    | otherwise          = Vec $ V.fromList [unsafeIndex mat (r, c) | r <- [0 .. rows - 1]]

-- | Extracts row from @Mat@.
safeIndexRow :: Mat a -> Int -> Maybe (Vec a)
safeIndexRow mat@(Mat rows cols _ _ _ _ x) r
    | r < 0 || r >= rows = Nothing
    | isTransposed mat   = Just $ Vec $ V.fromList [unsafeIndex mat (r, c) | c <- [0 .. cols - 1]]
    | otherwise          = Just $ Vec $ V.slice (indexMatToVec mat (r, 0)) cols x

-- | Extracts column from @Mat@.
safeIndexCol :: Mat a -> Int -> Maybe (Vec a)
safeIndexCol mat@(Mat rows cols _ _ _ _ x) c
    | c < 0 || c >= cols = Nothing
    | isTransposed mat   = Just $ Vec $ V.slice (indexMatToVec mat (0, c)) rows x
    | otherwise          = Just $ Vec $ V.fromList [unsafeIndex mat (r, c) | r <- [0 .. rows - 1]]


-- | Extracts diagonal from @Mat@.
diagonal :: Mat a -> Vec a
diagonal x = Vec $ V.fromList [unsafeIndex x (n, n) | n <- [0 .. min (nRows x) (nCols x) - 1]]

-- | Flattens @Mat@ to a @Vec@.
flatten :: Mat a -> Vec a
flatten = Vec . storage . force


-- Combining

-- | Applies function to every element of @Mat@.
map :: (a -> b) -> Mat a -> Mat b
map = fmap

-- | Applies function to a given row. If new @Vec@ is longer then кщц, it is truncated.
mapRow :: Int -> (Vec a -> Vec a) -> Mat a -> Mat a
mapRow row f mat = let newRow = f $ indexRow mat row
                   in imap (\(r, c) x -> if r == row then unsafeIndex newRow c else x) mat

-- | Applies function to a given column. If new @Vec@ is longer then column, it is truncated.
mapCol :: Int -> (Vec a -> Vec a) -> Mat a -> Mat a
mapCol col f mat = let newCol = f $ indexCol mat col
                   in imap (\(r, c) x -> if c == col then unsafeIndex newCol r else x) mat

-- | Flipped @map@.
for :: Mat a -> (a -> b) -> Mat b
for = flip map

-- | Applies function to every element and its position of @Mat@.
imap :: ((Int, Int) -> a -> b) -> Mat a -> Mat b
imap f mat@(Mat rows cols rk ck r0 c0 x) = Mat rows cols rk ck r0 c0 (V.imap (f . indexVecToMat mat) x)

-- | Zips two @Mat@s together using given function.
elementwise :: (a -> b -> c) -> Mat a -> Mat b -> Mat c
elementwise f a b
    | size a /= size b = error "Two matrices have different sizes"
    | otherwise        = let (rows, cols) = size a
                         in Mat rows cols cols 1 0 0 $
                            V.fromList [f (unsafeIndex a (r, c)) (unsafeIndex b (r, c)) | r <- [0 .. rows - 1], c <- [0 .. cols - 1]]


-- Operations with matrices

-- | Sets new size for a matrix relative to top left corner and uses given element for new entries if the matrix is extended.
setSize :: Mat a -> a -> (Int, Int) -> Mat a
setSize mat x = flip generate $ \(r, c) -> if r < nRows mat && c < nCols mat then unsafeIndex mat (r, c) else x

-- | Extends matrix size relative to top left corner using given element for new entries. The matrix is never reduced in size.
extend :: Mat a -> a -> (Int, Int) -> Mat a
extend mat x (rows, cols) = setSize mat x (max (nRows mat) rows, max (nCols mat) cols)

-- | Shrinks matrix size relative to top left corner. The matrix is never extended in size.
shrink :: Mat a -> (Int, Int) -> Mat a
shrink mat (rows, cols) = setSize mat undefined (min (nRows mat) rows, min (nCols mat) cols)


-- | Swaps two rows.
swapRows :: Mat a -> Int -> Int -> Mat a
swapRows mat@(Mat rows cols _ _ _ _ _) row1 row2
    | row1 < 0 || row2 < 0 || row1 >= rows || row2 >= rows = error "Given row indices are out of bounds"
    | otherwise                                            = generate (rows, cols) $ \(r, c) -> if r == row1 then unsafeIndex mat (row2, c)
                                                                                                else if r == row2 then unsafeIndex mat (row1, c)
                                                                                                else unsafeIndex mat (r, c)
-- | Swaps two columns.
swapCols :: Mat a -> Int -> Int -> Mat a
swapCols mat@(Mat rows cols _ _ _ _ _) col1 col2
    | col1 < 0 || col2 < 0 || col1 >= cols || col2 >= cols = error "Given column indices are out of bounds"
    | otherwise                                            = generate (rows, cols) $ \(r, c) -> if c == col1 then unsafeIndex mat (r, col2)
                                                                                                else if c == col2 then unsafeIndex mat (r, col1)
                                                                                                else unsafeIndex mat (r, c)


-- Submatrices

-- | Extacts minor matrix, skipping given row and column.
minor :: Mat a -> (Int, Int) -> Mat a
minor mat@(Mat rows cols _ _ _ _ _) (r', c') = Mat (rows - 1) (cols - 1) (cols - 1) 1 0 0 $
                                             V.fromList [unsafeIndex mat (r, c) | r <- [0 .. rows - 1], c <- [0 .. cols - 1], r /= r' && c /= c']

-- | Extracts submatrix, that is located between given two positions.
submatrix :: Mat a -> ((Int, Int), (Int, Int)) -> Mat a
submatrix (Mat rows cols rk ck r0 c0 x) ((r1, c1), (r2, c2))
    | r1 < 0 || c1 < 0 || r2 < 0 || c2 < 0 ||
      r2 < r1 || r2 > rows || c2 < c2 || c2 > cols = error "Given row and column limits are incorrect"
    | otherwise                                    = Mat (r2 - r1) (c2 - c1) rk ck (r0 + r1) (c0 + c1) x

-- | Splits matrix into 4 parts, given position is a pivot, that corresponds to first element of bottom-right subpart.
split :: Mat a -> (Int, Int) -> (Mat a, Mat a, Mat a, Mat a)
split x (r, c) = (submatrix x ((0, 0), (r, c)),       submatrix x ((0, c), (r, nCols x)),
                  submatrix x ((r, 0), (nRows x, c)), submatrix x ((r, c), (nRows x, nCols x)))

-- | Joins 4 blocks of matrices.
join :: (Mat a, Mat a, Mat a, Mat a) -> Mat a
join (tl@(Mat rowsTL colsTL _ _ _ _ _), tr@(Mat rowsTR colsTR _ _ _ _ _),
      bl@(Mat rowsBL colsBL _ _ _ _ _), br@(Mat rowsBR colsBR _ _ _ _ _))
    | rowsTL /= rowsTR || rowsBL /= rowsBR ||
      colsTL /= colsBL || colsTR /= colsBR = error "Matrices dimensions do not match"
    | otherwise                            = generate (rowsTL + rowsBL, colsTL + colsTR) $
                                             \(r, c) -> uncurry unsafeIndex $
                                                        if r >= rowsTL && c >= colsTL then (br, (r - rowsTL, c - colsTL))
                                                        else if r >= rowsTL           then (bl, (r - rowsTL, c))
                                                        else if c >= colsTL           then (tr, (r, c - colsTL))
                                                        else                               (tl, (r, c))

infixl 9 <|>
-- | Joins two matrices horizontally.
(<|>) :: Mat a -> Mat a -> Mat a
(<|>) a@(Mat rows1 cols1 _ _ _ _ _) b@(Mat rows2 cols2 _ _ _ _ _)
    | rows1 /= rows2 = error "Given matrices must have the same number of rows"
    | otherwise      = generate (rows1, cols1 + cols2) $ \(r, c) -> if c < cols1
                                                                    then unsafeIndex a (r, c)
                                                                    else unsafeIndex b (r, c - cols1)

infixl 9 <->
-- | Joins two matrices vertically.
(<->) :: Mat a -> Mat a -> Mat a
(<->) a@(Mat rows1 cols1 _ _ _ _ _) b@(Mat rows2 cols2 _ _ _ _ _)
    | cols1 /= cols2 = error "Given matrices must have the same number of columns"
    | otherwise      = generate (rows1 + rows2, cols1) $ \(r, c) -> if r < rows1
                                                                    then unsafeIndex a (r, c)
                                                                    else unsafeIndex b (r - rows1, c)


-- Functions that work on mathematical matrix (type constraint refers to a number)

-- | Creates @Mat@ that is filled with zeroes.
zeroes :: Num a => (Int, Int) -> Mat a
zeroes = flip replicate 0

-- | Creates @Mat@ that is filled with ones.
ones :: Num a => (Int, Int) -> Mat a
ones = flip replicate 1

-- | Creates identity matrix.
identity :: Num a => Int -> Mat a
identity n = generate (n, n) $ \(r, c) -> if r == c then 1 else 0


-- | Adamar multiplication (elementwise multiplication).
adamarMul :: Num a => Mat a -> Mat a -> Mat a
adamarMul = elementwise (*)

instance Num a => MatOps (Mat a) a where
    transpose (Mat rows cols rk ck r0 c0 x) = Mat cols rows ck rk c0 r0 x
    matMul a@(Mat rows1 cols1 _ _ _ _ _) b@(Mat rows2 cols2 _ _ _ _ _)
        | cols1 /= rows2 = error "Matrices dimensions do not match"
        | otherwise      = generate (rows1, cols2) $ \(r, c) -> SV.unSingleton $ indexRow a r `SV.dot` indexCol b c

-- | Determinant of a square matrix. If matrix is empty, zero is returned.
det :: Num a => Mat a -> a
det mat@(Mat rows cols _ _ _ _ _)
    | rows /= cols       = error "Matrix is not square"
    | nElements mat == 0 = 0
    | nElements mat == 1 = unsafeIndex mat (0, 0)
    | otherwise          = let mat' = if isTransposed mat then transpose mat else mat

                               determinant :: Num a => Mat a -> a
                               determinant x@(Mat 2 _ _ _ _ _ _) = unsafeIndex x (0, 0) * unsafeIndex x (1, 1) -
                                                                   unsafeIndex x (1, 0) * unsafeIndex x (0, 1)
                               determinant x                     = sum [((-1) ^ i) * (indexRow x 0 ! i) * determinant (minor x (0, i))
                                                                      | i <- [0 .. nRows x - 1]]
                           in determinant mat'

-- | Row reduced echelon form of matrix.
rref :: (Eq a, Fractional a) => Mat a -> Mat a
rref mat@(Mat rows cols _ _ _ _ _) = go mat 0 [0 .. rows - 1]
  where
    go m _ [] = m
    go m lead (r:rs) = case find ((0 /=) . unsafeIndex m) [(i, j) | j <- [lead .. cols - 1], i <- [r .. rows - 1]] of
                           Nothing                -> m
                           Just (pivotRow, lead') -> let newRow = SV.map (/ unsafeIndex m (pivotRow, lead')) (indexRow m pivotRow)
                                                         m'   = swapRows m pivotRow r
                                                         m''  = mapRow r (const newRow) m'
                                                         m''' = imap (\(row, c) -> if row == r
                                                                                   then id
                                                                                   else subtract (newRow ! c * unsafeIndex m'' (row, lead'))
                                                                     ) m''
                                                     in go m''' (lead' + 1) rs

-- | Inverse of a square matrix. If given matrix is empty, empty matrix is returned.
inverse :: (Eq a, Fractional a) => Mat a -> Maybe (Mat a)
inverse mat@(Mat rows cols _ _ _ _ _)
    | rows /= cols       = error "Matrix is not square"
    | nElements mat == 0 = Just empty
    | otherwise          = let mat' = mat <|> identity rows
                               reduced = rref mat'
                               (left, right, _, _) = split reduced (rows, cols)
                           in case V.find (== 0) (SV.unVec $ diagonal left) of
                                  Nothing -> Just right
                                  Just _  -> Nothing

-- | Orthogonalizes matrix by rows using Gram-Schmidt algorithm.
orthogonalized :: Floating a => Mat a -> Mat a
orthogonalized mat = foldl' (\mat' row -> mapRow row SV.normalized $ iterationGramSchmidt mat' row) mat [0 .. nRows mat]
  where
    iterationGramSchmidt mat' row = foldl' (\mat'' r ->
                                            mapRow row (subtract $ indexRow mat'' r *. SV.unSingleton (indexRow mat'' r `SV.dot` indexRow mat'' row)) mat''
                                           ) mat' [0 .. row]
