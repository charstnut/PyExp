from julia import Base
import julia
jl = julia.Julia()

jl.eval('include("helloworld.jl")')
foo = jl.eval('foo')
result = foo([1, 2, 3], [2, 3, 4])
print("julia evaluated function: type{}, result {}".format(
    type(result), result))
result2 = Base.Math.hypot(1, 2)**2
print("Using std lib in julia: type{}, result {}".format(
    type(result2), result2))
