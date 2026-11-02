module TensorTest.Ops.Convolution.Reference (reference) where

reference :: (Num a) => [Int] -> [Int] -> [Int] -> [Int] -> [a] -> [a] -> [a]
reference [batch, cin, height, width] [cout, _, kh, kw] [sy, sx] [top, bottom, left, right] input weights =
  [ sum
      [ input !! (((b * cin + c) * height + iy) * width + ix) * weights !! (((o * cin + c) * kh + ky) * kw + kx)
        | c <- [0 .. cin - 1],
          ky <- [0 .. kh - 1],
          let iy = y * sy + ky - top,
          iy >= 0,
          iy < height,
          kx <- [0 .. kw - 1],
          let ix = x * sx + kx - left,
          ix >= 0,
          ix < width
      ]
    | b <- [0 .. batch - 1],
      o <- [0 .. cout - 1],
      y <- [0 .. outH - 1],
      x <- [0 .. outW - 1]
  ]
  where
    outH = 1 + (height + top + bottom - kh) `div` sy
    outW = 1 + (width + left + right - kw) `div` sx
reference _ _ _ _ _ _ = error "Invalid reference convolution dimensions"
