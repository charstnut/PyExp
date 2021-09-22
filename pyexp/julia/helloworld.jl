λ₀ = 10
println("Hello world!")
println("λ₀ is $(λ₀)")
println(ARGS)
println("Chained expr: $(1 < 3 < 4 >= 4)")

function foo(x, y)
    return x.^2  + y.^2
end
