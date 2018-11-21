module QRUpdate

using LinearAlgebra
using LinearAlgebra.BLAS: gemv!, axpy!

abstract type OrthogonalizationMethod end

"""
    DGKS(r, correction, steps = 2)

Use the repeated, classical Gram-Schmidt method and store the projection in `r`.

Needs a pre-allocated temporary vector `correction` that is similar to `r`.

Will use at most `steps` applications of (I - VV').
"""
struct DGKS{TVr<:AbstractVector,TVc<:AbstractVector} <: OrthogonalizationMethod
    r::TVr
    correction::TVc
    steps::Int

    DGKS(r::TVr, correction::TVc, steps = 2) where {TVr<:AbstractVector,TVc<:AbstractVector} = new{TVr,TVc}(r, correction, steps)
end

"""
    ClassicalGramSchmidt(r)

Use the classical Gram-Schmidt method and store the projection in `r`
"""
struct ClassicalGramSchmidt{TV<:AbstractVector} <: OrthogonalizationMethod 
    r::TV
end

"""
    ModifiedGramSchmidt(r)

Use the modified Gram-Schmidt method and store the projection in `r`
"""
struct ModifiedGramSchmidt{TV<:AbstractVector} <: OrthogonalizationMethod 
    r::TV
end

"""
    orthogonalize_and_normalize!(V, w, method::OrthogonalizationMethod) → norm

Orthogonalize `w` in-place against the columns of `V`.

In exact arithmetic: `w ← (I - VV')w`. In finite precision rounding errors can occur.
Depending on the use case one might choose a different methods to orthogonalize a vector.

Often in literature the `ModifiedGramSchmidt` method is advocated (in iterative solvers 
like GMRES for instance). However, rounding errors can build up in modified Gram-Schmidt, 
so a stable alternative would be repeated Gram-Schmidt. Usually 'twice is enough' in the
sense that `w ← (I - VV')w` is performed twice: the second application of `(I - VV')` 
might remove the rounding errors of the first application. If `w` is nearly in the span of 
`V` more application might be necessary.

List of methods:
- `ModifiedGramSchmidt`: quite stable, BLAS1
- `ClassicalGramSchmidt`: very unstable, BLAS2
- `DGKS`: stable, BLAS2, ~ twice as much work as `ModifiedGramSchmidt` but usually fast.

"""
function orthogonalize_and_normalize!(V::AbstractMatrix{T}, w::AbstractVector{T}, p::DGKS) where {T}
    # Orthogonalize using BLAS-2 ops
    mul!(p.r, V', w)
    gemv!('N', -one(T), V, p.r, one(T), w)
    nrm = norm(w)

    # 1 / √2 is used in ARPACK
    η = inv(√(real(T)(2)))

    projection_size = norm(p.r)

    # Repeat as long as the DGKS condition is satisfied
    # Typically this condition is true only once.
    # todo: make this terminate after at most N steps.
    for i = Base.OneTo(p.steps - 1)
        (nrm > η * projection_size) && break
        mul!(p.correction, V', w)
        projection_size = norm(p.correction)
        # w = w - V * correction
        gemv!('N', -one(T), V, p.correction, one(T), w)
        axpy!(one(T), p.correction, p.r)
        nrm = norm(w)
    end

    # Normalize; note that we already have norm(w).
    rmul!(w, inv(nrm))

    nrm
end

function orthogonalize_and_normalize!(V::AbstractMatrix{T}, w::AbstractVector{T}, p::ClassicalGramSchmidt) where {T}
    # Orthogonalize using BLAS-2 ops
    mul!(p.r, V', w)
    gemv!('N', -one(T), V, p.r, one(T), w)
    nrm = norm(w)

    # Normalize
    rmul!(w, inv(nrm))

    nrm
end

function orthogonalize_and_normalize!(V::AbstractVector{<:AbstractVector{T}}, w::AbstractVector{T}, p::ModifiedGramSchmidt) where {T}
    # Orthogonalize using BLAS-1 ops
    for i = 1 : length(V)
        p.r[i] = dot(V[i], w)
        axpy!(-p.r[i], V[i], w)
    end

    # Normalize
    nrm = norm(w)
    rmul!(w, inv(nrm))

    nrm
end

function orthogonalize_and_normalize!(V::AbstractMatrix{T}, w::AbstractVector{T}, p::ModifiedGramSchmidt) where {T}
    # Orthogonalize using BLAS-1 ops and column views.
    for i = 1 : size(V, 2)
        column = view(V, :, i)
        p.r[i] = dot(column, w)
        axpy!(-p.r[i], column, w)
    end

    nrm = norm(w)
    rmul!(w, inv(nrm))

    nrm
end


end
