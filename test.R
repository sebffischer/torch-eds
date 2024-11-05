library(torch)

mod = nn_module("mod",
  initialize = function() {
    self$a = nn_parameter(torch_randn(1))
    self$b = nn_buffer(torch_randn(1))
  },
  forward = function(x) {
    self$a * x + b
  }
)

m = mod()
m$pa