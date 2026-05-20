"""

Package `SimpleExpressions` provide means to manage simple expressions.

"""
module SimpleExpressions

export ScaledVariable, LinearCombination

"""
    ex = ScaledVariable([α = 1,] sym)

yields an object representing the expression `α*sym` where `α` is a numerical
multiplier and `sym` is a variable name (a symbol or a string).

It is also possible to build an instance of `ScaledVariable` from a simple
expression like `:(a)`, `:(+a)`, `:(+a)`, `:(-a)`, `:(2a)`, `:(a/3)`, or
`:(5\a)`.

Instances of `ScaledVariable` can be negated, multiplied, or divided by a
numerical scalar using the usual Julia syntax: `+ex`, `-ex`, `β*ex`, `β\ex`, or
`ex/β`.

"""
struct ScaledVariable
    mult::Number
    name::Symbol
end

ScaledVariable(name::Union{Symbol,AbstractString}) =
    ScaledVariable(1, name)

ScaledVariable(mult::Real, name::AbstractString) =
    ScaledVariable(mult, Symbol(name))

function ScaledVariable(ex::Expr)
    res = try_convert(ScaledVariable, ex)
    res === nothing && argument_error("expression `", ex, "` is too complex")
    return res
end

Base.:(+)(ex::ScaledVariable) = ex
Base.:(-)(ex::ScaledVariable) = ScaledVariable(-ex.mult, ex.name)
Base.:(*)(ex::ScaledVariable, α::Real) = α*ex
Base.:(*)(α::Real, ex::ScaledVariable) = ScaledVariable(α*ex.mult, ex.name)
Base.:(/)(ex::ScaledVariable, α::Real) = ScaledVariable(ex.mult/α, ex.name)
Base.:(\)(α::Real, ex::ScaledVariable) = ex/α

function Base.convert(::Type{Expr}, var::ScaledVariable)
    if var.mult == 1
        return var.name
    elseif var.mult == -1
        return Expr(:call, :(-), var.name)
    else
        return Expr(:call, :(*), var.mult, var.name)
    end
end

Base.show(io::IO, ::MIME"text/plain", ex::ScaledVariable) = show(io, ex)
Base.show(io::IO, ex::ScaledVariable) =
    print(io, "ScaledVariable(:(", convert(Expr, ex), "))")

"""
    ex = LinearCombination(args...)

yields a simple expression which is a linear combination of variables.  Each
argument `args...` is pushed into an, initially empty, expression which is
returned.  The result is similar to sum of all arguments.

Each argument can be an expression that can be represented as a linear
combination of variables.

"""
struct LinearCombination <: AbstractVector{ScaledVariable}
    terms::Vector{ScaledVariable}
    LinearCombination(A::AbstractVector{ScaledVariable}) = new(A)
end

terms(A::LinearCombination) = getfield(A, :terms)

LinearCombination() = LinearCombination(ScaledVariable[])
LinearCombination(ex) = push!(LinearCombination(), ex)
LinearCombination(args...) = push!(LinearCombination(), args...)

Base.convert(::Type{T}, A::T) where {T<:LinearCombination} = A
Base.convert(::Type{T}, A) where {T<:LinearCombination} = T(A)

function Base.convert(::Type{Expr}, A::LinearCombination)
    expr = Expr(:call)
    args = expr.args
    for arg in A
        arg.mult == 0 && continue
        length(args) == 1 && prepend!(args, (:(+),))
        push!(args, convert(Expr, arg))
    end
    if length(args) == 0
        return 0
    elseif length(args) == 1
        return args[1]
    else
        return expr
    end
end

Base.show(io::IO, ::MIME"text/plain", A::LinearCombination) = show(io, A)
Base.show(io::IO, A::LinearCombination) =
    print(io, "LinearCombination(:(", convert(Expr, A), "))")

# Make simple expressions behave like vectors.
for fn in (:axes, :length, :size, :strides)
    @eval Base.$fn(A::LinearCombination) = $fn(terms(A))
end
Base.IndexStyle(::Type{<:LinearCombination}) = IndexStyle(Vector)
@inline function Base.getindex(A::LinearCombination, I...)
    B = terms(A)
    @boundscheck checkbounds(B, I...)
    @inbounds r = getindex(B, I...)
    return r
end
@inline function Base.setindex(A::LinearCombination, x, I...)
    B = terms(A)
    @boundscheck checkbounds(B, I...)
    @inbounds setindex!(B, x, I...)
    return A
end

"""
    push!(A::LinearCombination, ex...) -> A

pushes simple expressions `ex...` into the linear combination of variables `A`,
the result is similar as summing all arguments.

"""
function Base.push!(A::LinearCombination, args...)
    for ex in args
        push!(A, ex)
    end
    return A
end

function Base.push!(A::LinearCombination, ex::ScaledVariable)
    for k in eachindex(A.terms)
        if A.terms[k].name == ex.name
            A.terms[k] = ScaledVariable(ex.mult + A.terms[k].mult, ex.name)
            return A
        end
    end
    push!(A.terms, ex)
    return A
end

Base.push!(A::LinearCombination, name::Union{Symbol,AbstractString}) =
    push!(A, ScaledVariable(name))

function Base.push!(A::LinearCombination, ex::Expr; mult::Real=1)
    n = length(ex.args)
    if ex.head == :call
        fn = ex.args[1]
        if fn == :(+)
            for i in 2:n
                push!(A, mult*ScaledVariable(ex.args[i]))
            end
            return A
        elseif fn == :(-) && (n == 2 || n == 3)
            push!(A, mult*ScaledVariable(ex.args[2]))
            if n > 2
                push!(A, -mult*ScaledVariable(ex.args[3]))
            end
            return A
        else
            cnv = try_convert(ScaledVariable, ex)
            if cnv !== nothing
                return push!(A, cnv)
            end
        end
    end
    argument_error("expression `", ex, "` is too complex")
end

"""
    merge!(A::LinearCombination, B::LinearCombination...) -> A

merges simple linear combinations of variables represented by `A` and each of
`B...` into `A` and returns `A`.  The result is similar to summing all
arguments.

"""
function Base.merge!(A::LinearCombination, args::LinearCombination...)
    for B in args
        merge!(A, B)
    end
    return A
end

function Base.merge!(A::LinearCombination, B::LinearCombination)
    for ex in B.terms
        push!(A, ex)
    end
    return A
end

"""
    merge(args::LinearCombination...) -> A

merges all linear combinations of variables specified as arguments and return
an instance of `LinearCombination` which is similar to summing all arguments.

"""
function Base.merge(A::LinearCombination, args::LinearCombination...)
    merge!(copy(A), args...)
end

"""
    copy(A::LinearCombination) -> B

yields an independent copy of the linear combination of variables represented
by `A`.

"""
function Base.copy(A::LinearCombination)
    B = LinearCombination()
    n = length(A.terms)
    n ≥ 1 && copyto!(resize!(B.terms, n), 1, A.terms, 1, n)
    return B
end

@noinline argument_error(args...) = argument_error(string(args...))
argument_error(mesg::AbstractString) = throw(ArgumentError(mesg))

"""
    try_convert(ScaledVariable, ex)

attempts to convert expression `ex` into a term consisting in a numerical
multiplier and a symbolic name.  The result is `nothing` if `ex` is too
complex, an instance of `ScaledVariable` otherwise.

Simple expressions are like `:(a)`, `:(+a)`, `:(+a)`, `:(-a)`, `:(2a)`,
`:(a/3)`, or `:(5\a)`.

"""
try_convert(::Type{<:ScaledVariable}, ex::ScaledVariable) = ex

try_convert(::Type{<:ScaledVariable}, sym::Symbol) =
    ScaledVariable(sym)

function try_convert(::Type{<:ScaledVariable}, ex::Expr)
    n = length(ex.args)
    if ex.head == :call
        fn = ex.args[1]
        if fn === (:+)
            if n == 2 && isa(ex.args[2], Symbol)
                return ScaledVariable(ex.args[2])
            end
        elseif fn === (:-)
            if n == 2 && isa(ex.args[2], Symbol)
                return ScaledVariable(-1, ex.args[2])
            end
        elseif fn === (:*)
            if n == 3 && isa(ex.args[2], Number) && isa(ex.args[3], Symbol)
                return ScaledVariable(ex.args[2], ex.args[3])
            end
            if n == 3 && isa(ex.args[2], Symbol) && isa(ex.args[3], Number)
                return ScaledVariable(ex.args[3], ex.args[2])
            end
        elseif fn == (:\)
            if n == 3 && isa(ex.args[2], Number) && isa(ex.args[3], Symbol)
                return ScaledVariable(1/ex.args[2], ex.args[3])
            end
        elseif fn == (:/)
            if n == 3 &&  isa(ex.args[2], Symbol) && isa(ex.args[3], Number)
                return ScaledVariable(1/ex.args[3], ex.args[2])
            end
        end
    end
    return nothing
end

end # module
