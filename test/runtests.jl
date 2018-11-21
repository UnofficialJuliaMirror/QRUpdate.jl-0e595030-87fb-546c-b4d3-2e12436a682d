using QRUpdate: orthogonalize_and_normalize!, DGKS, ClassicalGramSchmidt, ModifiedGramSchmidt
using Test, Random, LinearAlgebra

@testset "Orthogonalization" begin

Random.seed!(1234321)
n = 10
m = 3

"""
Test whether w is w_original orthonormalized w.r.t. V,
given the projection h = V' * h and the norm of V * V' * h
"""
function is_orthonormalized(V::Matrix{T}, w_original, w, h, nrm) where {T}
    # Normality
    @test norm(w) ≈ one(real(T))

    # Orthogonality
    @test norm(V'w) ≈ zero(real(T)) atol = 10eps(real(T))

    # Denormalizing and adding the components in V should give back the original
    @test nrm * w + V * h ≈ w_original
end

@testset "Eltype $T" for T = (ComplexF32, Float64)

    # Create an orthonormal matrix V
    F = qr(rand(T, n, m))
    V = Matrix(F.Q)

    # And a random vector to be orth. to V.
    w_original = rand(T, n)

    # Assuming V is a matrix
    @testset "Using $method" for method = (ClassicalGramSchmidt, ModifiedGramSchmidt)

        # Projection size
        h = zeros(T, m)

        # Orthogonalize w in-place
        w = copy(w_original)
        nrm = orthogonalize_and_normalize!(V, w, method(h))

        is_orthonormalized(V, w_original, w, h, nrm)
    end

    @testset "Using DGKS" begin
        h = zeros(T, m)
        correction = zeros(T, m)
        w = copy(w_original)
        nrm = orthogonalize_and_normalize!(V, w, DGKS(h, correction))
        is_orthonormalized(V, w_original, w, h, nrm)
    end

    # Assuming V is a vector.
    @testset "ModifiedGramSchmidt with vectors" begin
        V_vec = [V[:, i] for i = 1 : m]

        # Projection size
        h = zeros(T, m)

        # Orthogonalize w in-place
        w = copy(w_original)
        nrm = orthogonalize_and_normalize!(V_vec, w, ModifiedGramSchmidt(h))

        is_orthonormalized(V, w_original, w, h, nrm)
    end
end
end
