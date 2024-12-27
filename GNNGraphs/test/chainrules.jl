@testitem "dict constructor" setup=[GraphsTestModule] begin
    using .GraphsTestModule
    grad = gradient(1.) do x
        d = Dict([:x => x, :y => 5]...)    
        return sum(d[:x].^2)
    end[1]

    @test grad == 2

    grad = gradient(1.) do x
        d = Dict([:x => x, :y => 5])    
        return sum(d[:x].^2)
    end[1]

    @test grad == 2

    grad = gradient(1.) do x
        d = Dict(:x => x, :y => 5)    
        return sum(d[:x].^2)
    end[1]

    @test grad == 2
end
