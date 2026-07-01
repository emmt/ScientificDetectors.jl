
default_valid_pixels_map(roi::DetectorAxes) = FastUniformArray(true, size(roi))

#------------------------------------------------------------------------------
# IDENTIFIERS

"""
    ScientificDetectors.Identifiers

Union of types acceptable for identifiers (strings, symbols, or integers).

"""
const Identifiers = Union{AbstractString,Symbol,Integer}

"""
    identifier(key) -> str

Converts `key` into a string identifier. Argument `key` can be of any type in union
`Identifiers` (a string, a symbol or an integer). Argument can be a tuple or an array of
identifiers.

"""
identifier(key::String) = key
identifier(key::AbstractString) = String(key)
identifier(key::Integer) = string("#",key)
identifier(key::Symbol) = String(key)
identifier(A::AbstractArray{String}) = A
identifier(A::AbstractArray{<:Identifiers}) = map(identifier, A)
identifier(A::Tuple{Vararg{String}}) = A
identifier(A::Tuple{Vararg{Identifiers}}) = map(identifier, A)

#------------------------------------------------------------------------------
#
# Sampler to provide samples given an array.
#
struct Sampler{T,N,Np1,A<:AbstractArray{T,Np1}}
    data::A
    inds::NTuple{N,Colon}
    function Sampler{T,N,Np1,A}(data::A) where {T,N,Np1,A<:AbstractArray{T,Np1}}
        Np1 == N + 1 || error("Np1 ≠ N + 1")
        Np1 ≥ 2 || error("insufficient number of dimensions")
        Base.has_offset_axes(data) && error(
            "data array has non-standard indexing")
        samples = size(data, Np1)
        samples ≥ 2 || error("insufficient number of samples")
        new{T,N,Np1,A}(data, colons(N))
    end
end

Sampler(data::A) where {T,N,A<:AbstractArray{T,N}} = Sampler{T,N-1,N,A}(data)

StatsBase.nobs(A::Sampler{T,N,Np1}) where {T,N,Np1} = size(A.data, Np1)

Base.eltype(A::Sampler{T,N}) where {T,N} = T
Base.ndims(A::Sampler{T,N}) where {T,N} = N
Base.length(A::Sampler) = nobs(A)
Base.size(A::Sampler) = size(A.data)[1:end-1]
Base.size(A::Sampler{T,N}, i::Integer) where {T,N} =
    (i < 1 ? error("out of range dimension index") :
     i ≤ N ? size(A.data, i) : 1)

Base.show(io::IO, obj::Sampler{T,N}) where {T,N} = begin
    join(io, size(obj),"×")
    print(io, " Sampler{$T,$N}: samples = ", nobs(obj))
end

Base.iterate(A::Sampler, i = 1) =
    (1 ≤ i ≤ nobs(A) ? (view(A.data, A.inds..., i), i+1) : nothing)

"""
    exposuretime(obj) -> Δt

yields the exposure time of object `obj`.

See also [`PreprocessingParameters`](@ref).

"""
function exposuretime end

"""
    getfields(x) -> (getfield(x,1), getfield(x,2), ..., getfield(x,nfields(x)))

yields a tuple of the fields of object `x`.

"""
getfields(x) = ntuple(i -> getfield(x, i), Val(nfields(x)))

"""
    nth(n)

yields the string `"\$n\$(ordinal_suffix(n))"`.

"""
nth(n::Integer) = string(n)*ordinal_suffix(n)
# NOTE: `string(n)*ordinal_suffix(n)` is about 3 times faster than
#       `string(n,ordinal_suffix(n))` or `"$n$(ordinal_suffix(n))"` which are
#       equally slow.

"""
    ordinal_suffix(n)

yields the ordinal suffix `"st"`, `"nd"`, `"rd"`, or `"th"` corresponding
to the value of the integer `n`.

"""
function ordinal_suffix(n::Integer)
    if n > zero(n)
        ten = oftype(n, 10)
        if rem(div(n, ten), ten) != one(n)
            # Number is positive and the tens digit is not 1.
            r = rem(n, ten)
            if r == oftype(r, 1)
                return "st"
            elseif r == oftype(r, 2)
                return "nd"
            elseif r == oftype(r, 3)
                return "rd"
            end
        end
    end
    return "th"
end

@noinline argument_error(args...) =
    argument_error(string(args...))
@noinline argument_error(mesg::AbstractString) =
    throw(ArgumentError(mesg))

@noinline dimension_mismatch() =
    dimension_mismatch("arguments have incompatible dimensions")
@noinline dimension_mismatch(args...) =
    dimension_mismatch(string(args...))
@noinline dimension_mismatch(mesg::AbstractString) =
    throw(DimensionMismatch(mesg))
