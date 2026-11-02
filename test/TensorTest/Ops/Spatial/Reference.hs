module TensorTest.Ops.Spatial.Reference (conv, transposed, patches, combine, pool, outputSizes, convWith, transposedWith, patchesWith, combineWith, poolWith) where

import Data.List (zipWith4)

points :: [Int] -> [[Int]]
points = mapM (\size -> [0 .. size - 1])

at :: [Int] -> [a] -> [Int] -> a
at dims values ix = values !! foldl (\linear (size, i) -> linear * size + i) 0 (zip dims ix)

outputSizes :: [Int] -> [Int] -> [Int] -> [Int] -> [Int]
outputSizes spatial kernel stride padding = zipWith4 (\size k s (lo, hi) -> 1 + (size + lo + hi - k) `div` s) spatial kernel stride $ pairs padding

pairs :: [a] -> [(a, a)]
pairs [] = []
pairs (a : b : rest) = (a, b) : pairs rest
pairs _ = error "Invalid reference padding"

valid :: [Int] -> [Int] -> Bool
valid sizes ix = and $ zipWith (\size i -> i >= 0 && i < size) sizes ix

source :: [Int] -> [Int] -> [Int] -> [Int] -> [Int]
source stride padding p = zipWith4 (\o s before q -> o * s - before + q) p stride (map fst $ pairs padding)

conv :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [a] -> [a] -> [a]
conv inputShape@(batch : cin : spatial) weightShape@(cout : _ : kernel) stride padding input weights =
  [ sum [at inputShape input (b : c : ix) * at weightShape weights (o : c : k) | c <- [0 .. cin - 1], k <- points kernel, let ix = source stride padding p k, valid spatial ix]
    | b <- [0 .. batch - 1],
      o <- [0 .. cout - 1],
      p <- points $ outputSizes spatial kernel stride padding
  ]
conv _ _ _ _ _ _ = error "Invalid reference convolution"

transposed :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> [a] -> [a] -> [a]
transposed inputShape@(batch : cin : spatial) weightShape@(_ : cout : kernel) stride padding extra input weights =
  [ sum [at inputShape input (b : c : ix) * at weightShape weights (c : o : k) | c <- [0 .. cin - 1], k <- points kernel, let shifted = zipWith3 (\i pad q -> i + pad - q) p before k, and $ zipWith (\i s -> i >= 0 && i `mod` s == 0) shifted stride, let ix = zipWith div shifted stride, valid spatial ix]
    | b <- [0 .. batch - 1],
      o <- [0 .. cout - 1],
      p <- points sizes
  ]
  where
    before = map fst $ pairs padding
    sizes = zipWith (+) extra $ zipWith4 (\size k s (lo, hi) -> (size - 1) * s - lo - hi + k) spatial kernel stride $ pairs padding
transposed _ _ _ _ _ _ _ = error "Invalid reference transpose"

patches :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [a] -> [a]
patches inputShape@(batch : cin : spatial) kernel stride padding input =
  [let ix = source stride padding p k in if valid spatial ix then at inputShape input (b : c : ix) else 0 | b <- [0 .. batch - 1], c <- [0 .. cin - 1], k <- points kernel, p <- points $ outputSizes spatial kernel stride padding]
patches _ _ _ _ _ = error "Invalid reference patches"

combine :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [a] -> [a]
combine (batch : cin : spatial) kernel stride padding values =
  [ sum [at [batch, cin * product kernel, product sizes] values [b, c * product kernel + ki, positionIndex] | (ki, k) <- zip [0 ..] $ points kernel, (positionIndex, p) <- zip [0 ..] $ points sizes, source stride padding p k == ix]
    | b <- [0 .. batch - 1],
      c <- [0 .. cin - 1],
      ix <- points spatial
  ]
  where
    sizes = outputSizes spatial kernel stride padding
combine _ _ _ _ _ = error "Invalid reference combination"

pool :: [Int] -> [Int] -> [Int] -> [Int] -> ([Double] -> Double) -> [Double] -> [Double]
pool inputShape@(batch : cin : spatial) kernel stride padding reduce input =
  [reduce [at inputShape input (b : c : ix) | k <- points kernel, let ix = source stride padding p k, valid spatial ix] | b <- [0 .. batch - 1], c <- [0 .. cin - 1], p <- points $ outputSizes spatial kernel stride padding]
pool _ _ _ _ _ _ = error "Invalid reference pooling"

outputWith :: [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> [Int]
outputWith spatial kernel stride padding dilation = outputSizes spatial (zipWith (\k d -> d * (k - 1) + 1) kernel dilation) stride padding

sourceWith :: [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> [Int]
sourceWith stride padding dilation p k = source stride padding p $ zipWith (*) dilation k

convWith :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> [a] -> [a] -> [a]
convWith inputShape@(batch : cin : spatial) weightShape@(cout : perInput : kernel) stride padding dilation groups input weights =
  [ sum [at inputShape input (b : ((o `div` perOutput) * perInput + c) : ix) * at weightShape weights (o : c : k) | c <- [0 .. cin `div` groups - 1], k <- points kernel, let ix = sourceWith stride padding dilation p k, valid spatial ix]
    | b <- [0 .. batch - 1],
      o <- [0 .. cout - 1],
      p <- points $ outputWith spatial kernel stride padding dilation
  ]
  where
    perOutput = cout `div` groups
convWith _ _ _ _ _ _ _ _ = error "Invalid grouped convolution reference"

transposedWith :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> Int -> [Int] -> [a] -> [a] -> [a]
transposedWith inputShape@(batch : cin : spatial) weightShape@(_ : perOutput : kernel) stride padding dilation groups extra input weights =
  [ sum [at inputShape input (b : ci : ix) * at weightShape weights (ci : (o `mod` perOutput) : k) | c <- [0 .. perInput - 1], let ci = (o `div` perOutput) * perInput + c, k <- points kernel, let shifted = zipWith3 (\i pad q -> i + pad - q) p before (zipWith (*) k dilation), and $ zipWith (\i s -> i >= 0 && i `mod` s == 0) shifted stride, let ix = zipWith div shifted stride, valid spatial ix]
    | b <- [0 .. batch - 1],
      o <- [0 .. perOutput * groups - 1],
      p <- points sizes
  ]
  where
    perInput = cin `div` groups
    before = map fst $ pairs padding
    effective = zipWith (\k d -> d * (k - 1) + 1) kernel dilation
    sizes = zipWith (+) extra $ zipWith4 (\size k s (lo, hi) -> (size - 1) * s - lo - hi + k) spatial effective stride $ pairs padding
transposedWith _ _ _ _ _ _ _ _ _ = error "Invalid grouped transpose reference"

patchesWith :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> [a] -> [a]
patchesWith inputShape@(batch : cin : spatial) kernel stride padding dilation input =
  [let ix = sourceWith stride padding dilation p k in if valid spatial ix then at inputShape input (b : c : ix) else 0 | b <- [0 .. batch - 1], c <- [0 .. cin - 1], k <- points kernel, p <- points $ outputWith spatial kernel stride padding dilation]
patchesWith _ _ _ _ _ _ = error "Invalid dilated patches reference"

combineWith :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> [a] -> [a]
combineWith (batch : cin : spatial) kernel stride padding dilation values =
  [sum [at [batch, cin * product kernel, product sizes] values [b, c * product kernel + ki, position] | (ki, k) <- zip [0 ..] $ points kernel, (position, p) <- zip [0 ..] $ points sizes, sourceWith stride padding dilation p k == ix] | b <- [0 .. batch - 1], c <- [0 .. cin - 1], ix <- points spatial]
  where
    sizes = outputWith spatial kernel stride padding dilation
combineWith _ _ _ _ _ _ = error "Invalid dilated combination reference"

poolWith :: [Int] -> [Int] -> [Int] -> [Int] -> [Int] -> ([Double] -> Double) -> [Double] -> [Double]
poolWith inputShape@(batch : cin : spatial) kernel stride padding dilation reduce input =
  [reduce [at inputShape input (b : c : ix) | k <- points kernel, let ix = sourceWith stride padding dilation p k, valid spatial ix] | b <- [0 .. batch - 1], c <- [0 .. cin - 1], p <- points $ outputWith spatial kernel stride padding dilation]
poolWith _ _ _ _ _ _ _ = error "Invalid dilated pooling reference"
