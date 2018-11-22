module QRUpdate

using LinearAlgebra
using LinearAlgebra.BLAS: gemv!, axpy!
using Base: OneTo

abstract type OrthogonalizationMethod end

"""
    DGKS(tmp, steps = 3)

Repeated classical Gram-Schmidt: most stable option, approximately twice or thee times as 
expensive as `ClassicalGramSchmidt()`. Needs a temporary vector to store Q'v in.

Will use at most `steps` applications of `(I - QQ')`.
"""
struct DGKS{TVc<:AbstractVector} <: OrthogonalizationMethod
    tmp::TVc
    steps::Int

    DGKS(tmp::TVc, steps = 2) where {TVc<:AbstractVector} = new{TVc}(tmp, steps)
end

"""
    ClassicalGramSchmidt()

Unstable but efficient way (BLAS2) to do orthogonalization
"""
struct ClassicalGramSchmidt <: OrthogonalizationMethod end

"""
    ModifiedGramSchmidt()

Quite stable but inefficient (BLAS1) way to do orthogonalization
"""
struct ModifiedGramSchmidt <: OrthogonalizationMethod end

"""
    orthogonalize_and_normalize!(Q, v, r, method::OrthogonalizationMethod) → norm
    orthogonalize_and_normalize!(Q, v, r, method::DGKS) → norm, success

Orthogonalize `v` in-place against the columns of `Q` and store `r ← Q' * v`.

In exact arithmetic: `v ← (I - QQ')v`. In finite precision rounding errors can occur.
Depending on the use case one might choose a different methods to orthogonalize a vector.

In the case of `DKGS` the second return value is a flag showing whether the 
orthogonalization succeeded in the number of steps provided. If `success = false` this can
be interpreted as `r` being in the column span of `Q`.

Often in literature the `ModifiedGramSchmidt` method is advocated (in iterative solvers 
like GMRES for instance). However, rounding errors can build up in modified Gram-Schmidt, 
so a stable alternative would be repeated Gram-Schmidt. Usually 'twice is enough' in the
sense that `v ← (I - QQ')v` is performed twice: the second application of `(I - QQ')` 
might remove the rounding errors of the first application. If `v` is nearly in the span of 
`Q` more application might be necessary.

List of methods:
- `ModifiedGramSchmidt()`: quite stable, BLAS1
- `ClassicalGramSchmidt()`: very unstable, BLAS2
- `DGKS(tmp,steps)`: stable, BLAS2, approximately two/three times the work 
  of `ClassicalGramSchmidt`
"""
function orthogonalize_and_normalize!(Q::AbstractMatrix{T}, v::AbstractVector{T}, r::AbstractVector{T}, p::DGKS) where {T}
    nrm = norm(v)
    fill!(r, zero(T))
    
    # used in ARPACK
    η = real(T)(1 / √2)

    for k = OneTo(p.steps)
        mul!(p.tmp, Q', v)
        gemv!('N', -one(T), Q, p.tmp, one(T), v)
        axpy!(one(T), p.tmp, r)
        prevnrm = nrm
        nrm = norm(v)

        if (nrm > η * prevnrm)
            rmul!(v, inv(nrm))
            return nrm, true
        end
    end

    rmul!(v, inv(nrm))
    return nrm, false
end

function orthogonalize_and_normalize!(Q::AbstractMatrix{T}, v::AbstractVector{T}, r::AbstractVector{T}, p::ClassicalGramSchmidt) where {T}
    # Orthogonalize using BLAS-2 ops
    mul!(r, Q', v)
    gemv!('N', -one(T), Q, r, one(T), v)
    nrm = norm(v)

    # Normalize
    rmul!(v, inv(nrm))

    nrm
end

function orthogonalize_and_normalize!(Q::AbstractVector{<:AbstractVector{T}}, v::AbstractVector{T}, r::AbstractVector{T}, p::ModifiedGramSchmidt) where {T}
    # Orthogonalize using BLAS-1 ops
    for i = 1 : length(Q)
        r[i] = dot(Q[i], v)
        axpy!(-r[i], Q[i], v)
    end

    # Normalize
    nrm = norm(v)
    rmul!(v, inv(nrm))

    nrm
end

function orthogonalize_and_normalize!(Q::AbstractMatrix{T}, v::AbstractVector{T}, r::AbstractVector{T}, p::ModifiedGramSchmidt) where {T}
    # Orthogonalize using BLAS-1 ops and column views.
    for i = 1 : size(Q, 2)
        column = view(Q, :, i)
        r[i] = dot(column, v)
        axpy!(-r[i], column, v)
    end

    nrm = norm(v)
    rmul!(v, inv(nrm))

    nrm
end


end
