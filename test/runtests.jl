using QRUpdate: orthogonalize_and_normalize!, DGKS, ClassicalGramSchmidt, ModifiedGramSchmidt
using Test, Random, LinearAlgebra

@testset "Orthogonalization" begin

Random.seed!(1234321)
n = 10
m = 3

"""
Test whether `v` is `v_original` orthonormalized w.r.t. Q,
given the projection `h = Q' * h` and the norm of `Q * Q' * h`
"""
function is_orthonormalized(Q::Matrix{T}, v_original, v, r, nrm) where {T}
    # Normality
    @test norm(v) ≈ one(real(T))

    # Orthogonality
    @test norm(Q' * v) ≈ zero(real(T)) atol = 10eps(real(T))

    # Denormalizing and adding the components in V should give back the original
    @test nrm * v + Q * r ≈ v_original
end

@testset "Eltype $T" for T = (ComplexF32, Float64)

    # Create an orthonormal matrix V
    Q = Matrix(qr(rand(T, n, m)).Q)

    # And a random vector to be orth. to V.
    v_original = rand(T, n)

    # Assuming V is a matrix
    @testset "Using $method" for method = (ClassicalGramSchmidt(), ModifiedGramSchmidt())

        # Projection size
        r = zeros(T, m)
        v = copy(v_original)
        nrm = orthogonalize_and_normalize!(Q, v, r, method)
        is_orthonormalized(Q, v_original, v, r, nrm)
    end

    @testset "Using DGKS" begin
        r = zeros(T, m)
        v = copy(v_original)
        nrm, success = orthogonalize_and_normalize!(Q, v, r, DGKS(zeros(T, m), 2))
        println(success)
        @test success
        is_orthonormalized(Q, v_original, v, r, nrm)
    end

    # Assuming Q is a vector.
    @testset "ModifiedGramSchmidt with vectors" begin
        Qvecs = [Q[:, i] for i = 1 : m]

        # Projection size
        r = zeros(T, m)

        # Orthogonalize w in-place
        v = copy(v_original)
        nrm = orthogonalize_and_normalize!(Qvecs, v, r, ModifiedGramSchmidt())

        is_orthonormalized(Q, v_original, v, r, nrm)
    end
end
end
