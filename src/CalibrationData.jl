"""
    A = CalibrationData{T}(roi, args...)

builds an empty instance of `CalibrationData` to collect statistics about
calibration data frames with the precision specified by the floating-point type
`T`, in the region of interest `roi`, and for calibration categories specified
by the remaining arguments.

The region of interest `roi` is an `N`-tuple of detector dimensions or
instances of `DetectorAxis`.

The type parameter `T` is the floating-point type for computations.  If
unspecified, `Float64` is assumed.

Calibrations categories have names and are (usually) simple sums of
contributions by a number of so-called *sources*, the sources are identified by
a symbolic name.  The simplest way to indicate this mapping is to call the
constructor as in the following example:

    A = CalibrationData{Float32}(roi,
                                 "DARK"  => :(dark),
                                 "LAMP1" => :(dark + lamp1),
                                 "LAMP2" => :(dark + lamp2),
                                 "LAMP1+LAMP2" => :(dark + lamp1 + lamp2), ...)

where there are 4 categories named `"DARK"`, `"LAMP1", `"LAMP2", and
`"LAMP1+LAMP2" which are relatedby the paired expressions to 3 sources named
`:dark', `:lamp1`, and `:lamp2`.  For instance, in the calibration named
`"LAMP1"`, the contributing sources are `:dark` and `:lamp1`.  There may be
numerical multipliers in the expressions but this is not needed in general.
The use of uppercase / lowercase letters in the above example is just to make
clear the distinction between calibration categories and sources, this is not
mandatory.  The above example can be understod as follow:

- There are 2 calibration lamps (the sources `:lamp1` and `:lamp2`) and `:dark`
  is to account for the contribution of dark current.

- There are 4 categories of calibration depending on whether each lamp is
  switched on or off and the dark current always contributes to calibration
  data.

In the assumed pixel model, the contributions of the source terms are always
multiplied by the exposure time.

The list of calibration categories may also be provided by a vector of
names-expressions terms of by instances of `CalibrationCategory`.

To add some calibration data frame(s) to `A`, call:

    push!(A, x...)

where each `x` is an instance of `CalibrationDataFrame`.

To push all calibration data frames produced by an iterator `itr`, just call:

    merge!(A, itr)

Other methods applicable to a `CalibrationData` instance `A`:

- `isempty(A)` yields whether any calibration data has been collected;

- `empty!(A)` discards all calibration data collected so far and returns `A`;

- `nobs(A)` yields the total number of collected data frames;

- `keys(A)` yields an iterable over the 2-tuples `(cat,Δt)` of categories and
  exposure times in collected data frames;

- `DetectorAxes(A)` yields the ROI.

"""
struct CalibrationData{T<:AbstractFloat,N}
    # All calibration data must have the same detector axes settings.
    roi::DetectorAxes{N}

    # Mapping between (cat,Δt) pairs and indices in the vector of collected
    # statistics.
    stat_index::Dict{Tuple{String,T},Int}

    # Vector of collected data statistics, each entry is for a given
    # (cat,Δt) pair.
    stat::Vector{IndependentStatistic{T,N,2}} # 2 statistical moments

    # Array shared by statistics of singleton data-sets to store their 2nd
    # moment.
    null::Array{T,N}

    # Source to category matrix.
    src_to_cat::Matrix{T}

    # Dictionary mapping categories to their index.
    cat_index::Dict{String,Int}

    # Dictionary mapping sources to their index.
    src_index::Dict{String,Int}
end

DetectorAxes{N}(cal::CalibrationData{T,N}) where {T,N} = DetectorAxes(cal)
DetectorAxes(cal::CalibrationData) = getfield(cal, :roi)

Base.eltype(A::CalibrationData) = eltype(typeof(A))
Base.eltype(::Type{<:CalibrationData{T}}) where {T} = T

Base.ndims(A::CalibrationData) = ndims(typeof(A))
Base.ndims(::Type{<:CalibrationData{T,N}}) where {T,N} = N

Base.size(A::CalibrationData) = size(DetectorAxes(A))
Base.size(A::CalibrationData, d::Integer) = size(DetectorAxes(A), d)

Base.axes(A::CalibrationData) = axes(DetectorAxes(A))
Base.axes(A::CalibrationData, d::Integer) = axes(DetectorAxes(A), d)

"""
    CalibrationCategory(catname, expr)

yields an object defining a calibration category whose name is `catname` and
which combine calibration sources as specified by `expr` which is a simple
linear combination of sources.  If argument `expr` is missing, a single source
with the same symbolic name as the calibration category is assumed.

Examples:

    CalibrationCategory("DARK", :(dark))
    CalibrationCategory("LAMP", :(dark + lamp1))
    CalibrationCategory("BACKGROUND", :(dark + background))
    CalibrationCategory("SKY", :(dark + background + sky))

"""
struct CalibrationCategory
    name::String            # name of category
    expr::LinearCombination # linear combination of sources
end

# FIXME: LinearCombination{K,V<:Number}
CalibrationCategory(name::AbstractString, ex::Expr) =
    CalibrationCategory(name, LinearCombination(ex))
CalibrationCategory(name::AbstractString, ex::ScaledVariable) =
    CalibrationCategory(name, LinearCombination(ex))
CalibrationCategory(name::AbstractString) =
    CalibrationCategory(name, ScaledVariable(name))

CalibrationCategory(kv::Pair) = CalibrationCategory(kv[1], kv[2])

for argtype in (:(Union{CalibrationCategory,Pair,AbstractString}...),
                :(Tuple{Vararg{Union{CalibrationCategory,
                                     Pair,AbstractString}}}),
                :(AbstractVector{<:Union{Pair,AbstractString}}))
    @eval function CalibrationData{T}(roi::DetectorAxes,
                                      cats::$argtype;
                                      kwds...) where {T<:AbstractFloat}
        return CalibrationData{T}(roi, collect(CalibrationCategory, cats);
                                  kwds...)
    end
end

function CalibrationData{T}(roi::DetectorAxes{N},
                            A::AbstractVector{CalibrationCategory};
                            verb::Bool=false) where {T<:AbstractFloat,N}

    # Compiles the information in `A` to build the ordered lists of calibration
    # categories `cats` and of calibration sources `srcs` and the calibration
    # sources to calibration categories matrix `H`.

    # Build the list of all source and calibration category names skipping
    # those with a null multiplier.
    srcs = Symbol[]
    cats = String[]
    for cal in A
        valid = false
        for ex in cal.expr
            if ex.mult > 0
                valid = true
                update_list!(srcs, ex.name)
            elseif ex.mult != 0
                error("multiplier for source \"", ex.name,
                      "\" must be non-negative")
            end
        end
        if valid
            update_list!(cats, cal.name)
        end
    end

    # Sort the lists.
    sort!(srcs)
    sort!(cats)
    if verb
        println("sources:    ", srcs)
        println("categories: ", cats)
    end

    # Build the sources to calibration categories matrix.
    nrows = length(cats)
    nrows ≥ 1 || argument_error("there must be some calibration categories")
    ncols = length(srcs)
    ncols ≥ 1 || argument_error("there must be some calibration sources")
    nrows ≥ ncols || argument_error(
        "there must be at least as many calibration categories as sources")
    H = fill!(Matrix{Float64}(undef, nrows, ncols), NaN)
    h = Vector{Float64}(undef, ncols) # to store a row of H
    for cal in A
        i = findfirst(x -> x == cal.name, cats)
        i === nothing && error(
            "missing calibration category \"", cal.name, "\"")
        compile!(h, cal.expr, srcs)
        first_time = true
        for j in 1:ncols
            first_time &= isnan(H[i,j])
        end
        if first_time
            for j in 1:ncols
                H[i,j] = h[j]
            end
        else
            for j in 1:ncols
                H[i,j] == h[j] || error(
                    "calibration category \"", cal.name, "\" defined ",
                    "with different combinations of sources")
            end
        end
    end
    rank(H) ≥ min(nrows, ncols) || argument_error(
        "calibration source to calibration category matrix is rank deficient")

    # Build calibration categories and sources dictionaries.
    cat_index = Dict{String,Int}()
    for (idx, str) in enumerate(cats)
        cat_index[str] = idx
    end
    src_index = Dict{String,Int}()
    for (idx, sym) in enumerate(srcs)
        str = String(sym)
        src_index[str] = idx
    end

    # Build instance.
    return CalibrationData{T,N}(
        roi,                               # region of interest
        Dict{Tuple{String,Float64},Int}(), # stat_index
        IndependentStatistic{T,N,2}[],     # stat (2 moments)
        zeros(T, size(roi)),               # null,
        H,                                 # src_to_cat
        cat_index,                         # cat_index
        src_index)                         # src_index
end

# Use double precision if type parameter is not specified.
CalibrationData(args...; kwds...) = CalibrationData{Float64}(args...; kwds...)

# Convert ROI.
#FIXME: seeems to loop in some cases
function CalibrationData{T}(roi::NTuple{N,DetectorAxisTypes},
                            args...; kwds...) where {T<:AbstractFloat,N}
    return CalibrationData{T}(DetectorAxes(roi), args...; kwds...)
end

"""
    compile(E::LinearCombination, S::AbstractVector{Symbol}) -> A

yields a list of multipliers for the symbolic names in `S` corresponding to the
linear combination of variables specified by `E`.  All variables specified in
`E` must be part of `S`.

"""
compile(E::LinearCombination, S::AbstractVector{Symbol}) =
    compile!(Vector{Float64}(undef, length(S)), A, S)

"""
    compile!(A::AbstractVector{<:Number}, E::LinearCombination,
             S::AbstractVector{Symbol}) -> A

stores in `A` the multipliers for the symbolic names in `S` corresponding to
the linear combination of variables specified by `E`.  All variables specified
in `E` must be part of `S`.  The destination `A` is returned.

"""
function compile!(A::AbstractVector{<:Number},
                  E::LinearCombination,
                  S::AbstractVector{Symbol})

    I = axes(A, 1)
    axes(S) == (I,) || error(
        "vectors of coefficients and of symbolic names have ",
        "incompatible indices")
    fill!(A, zero(eltype(A)))
    n = last(I)
    for ex in E.terms
        for i in I
            if ex.name == S[i]
                A[i] += ex.mult
                break
            end
            i < n || error("unexpected variable \"", ex.name, "\"")
        end
    end
    return A
end

"""
    update_list!(A, x) -> A

append `x` to vector `A` if not already in `A` and returns `A`.

"""
update_list!(A::AbstractVector{T}, x) where {T} = update_list!(A, as(T, x))

function update_list!(A::AbstractVector{T}, x::T) where {T}
    @inbounds for i in eachindex(A)
        if A[i] == x
            return A
        end
    end
    push!(A, x)
end

function Base.merge!(A::CalibrationData, itr)
    for x in itr
        push!(A, x)
    end
    return A
end

function Base.push!(A::CalibrationData, args...)
    for x in args
        push!(A, x)
    end
    return A
end

function Base.push!(A::CalibrationData{T,N},
                    x::CalibrationDataFrame{<:Real,N}) where {T<:AbstractFloat,N}
    # Extract and check fields.
    cat = category(x)
    haskey(A.cat_index, cat) || argument_error(
        "category\"", cat, "\" does not exists in calibration data")
    Δt = exposuretime(x)
    pxl = pixels(x)
    roi = DetectorAxes(x)
    Δt ≥ 0 || argument_error("exposure time must be nonnegative")
    Base.has_offset_axes(pxl) && argument_error(
        "array of pixels must have 1-based indices")
    dims = size(pxl)
    dims == size(roi) || dimension_mismatch(
        "array of pixels and region of interest have different sizes")
    roi == DetectorAxes(A) || argument_error(
        "detector ROI settings must be identical for all calibration data")

    # Update statistics for given category and exposure time.
    key = (cat, Δt)
    if !haskey(A.stat_index, key)
        # Create new instance of statistics
        push!(A.stat, IndependentStatistic(T, 2, dims)) # 2 statistical moments
        index = length(A.stat)
        A.stat_index[key] = index
    end
    weight = 1
    fit!(A.stat[A.stat_index[key]], pxl, weight)
    return A
end

@noinline Base.push!(A::CalibrationData, x::T) where {T} =
    argument_error(
        "Cannot convert argument(s) of type `$T` to calibration data frame ",
        "of type `CalibrationDataFrame`.  The solution may be to extend ",
        "`Base.push!(A::CalibrationData, x::$T)` for such kind of argument.")

Base.keys(A::CalibrationData) = keys(A.stat_index)
Base.isempty(A::CalibrationData) = isempty(A.stat_index)
Base.empty!(A::CalibrationData) = begin
    empty!(A.stat_index)
    empty!(A.stat)
    return A
end

StatsBase.nobs(A::CalibrationData) = begin
    n = 0
    for x in A.stat
        n += nobs(x)
    end
    return n
end

"""
    B = prunecalibration(A::CalibrationData{T,N}) where {T,N}

    Return a new CalibrationData `B` from CalibrationData `A` pruned of empty categories and sources
"""
function prunecalibration(A::CalibrationData{T,N}) where {T,N}
    src_nobs = zeros(length(A.src_index))
    existing_cat = falses(length(A.cat_index))
    nonzero_stat = falses(length(A.stat_index))
    new_src = Dict{String,Int}()
    new_cat = Dict{String,Int}()
    for (src, s) ∈ A.src_index
        for (cat, c) ∈ A.cat_index
            if (A.src_to_cat[c,s] ≠ 0)
                for (key, i) ∈ A.stat_index
                    if cat == key[1]             # category name
                        src_nobs[s] += nobs(A.stat[i])
                        existing_cat[c] |= true
                        new_src[src] = s
                        new_cat[cat] = c
                    end
                end
            end
        end
    end
    existing_src = src_nobs .≠0

    # update source to categories matrix
    H = A.src_to_cat[existing_cat,existing_src]

    # update categories dictionary

    ncat=cumsum(existing_cat)
    for (cat, c) ∈ A.cat_index
        if existing_cat[c]
            new_cat[cat] = ncat[c]
        end
    end

    # update sources dictionary
    nsrc=cumsum(existing_src)
    for (src, s) ∈ A.src_index
        if existing_src[s]
            new_src[src] = nsrc[s]
        end
    end

    # Build instance.
    return CalibrationData{T,N}(
        A.roi,                # region of interest
        A.stat_index,         # stat_index
        A.stat,               # stat
        A.null,               # null,
        H,                    # src_to_cat
        new_cat,              # cat_index
        new_src)              # src_index
end
