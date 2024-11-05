library(torch)

x = torch_randn(1000, 1000)
xt$is_contiguous()
xt = x$t()
xt$is_contiguous()

bench::mark(
  x + x,
  xt + x,
  check = FALSE,
  iterations = 1000
)
