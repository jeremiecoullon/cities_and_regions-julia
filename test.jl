#!/usr/bin/env julia

using Test
using DelimitedFiles

include("./src/Potential.jl")

"Run some basic tests"

cost_mat = readdlm("data/london_n/cost_mat.txt")
orig = readdlm("data/london_n/P.txt")

N, M = size(cost_mat)

xval = [0.35783767264345334, -0.2209858163853502, 0.07694316282590163, -0.5592493820167688,
0.45017948071235336, -0.13187935288390085, 0.08173972881194702, 0.4697604252725478, -0.7930020302715701,
 0.636736964081372, 0.04159245658542776, -0.5011748638445761, -0.5322225987871061, -0.766410957379569,
 0.958442692383763, 0.626177847931769, 0.610182863131767, 0.24255338002914106, -0.3249586529212869,
  0.8402797620935014, 0.9584105409783232, 0.3838693989980384, 0.2554425489860863, 0.009636452399528928,
  0.8633302667076528, 0.4971420481429969, 0.7620931640627635, -0.6214045967132642, 0.6885390087081147,
   -0.8332058405202001, -0.8445006788282816, 0.058018387092042545, 0.7448517737345446, -0.1102485355047067,
   0.6091525003404037, 0.4703273849893166, -0.3186939596967906, -0.6148175125420252, -0.6271015543044971,
    0.7662334229195285, 0.40033467437582004, -0.49266705127946153, 0.3845870729035221, 0.3956082166537964,
    0.9261066681673142, 0.054700333751995345, 0.16057794939771997, -0.23830634738546452, 0.6484556249254738]

# default theta in 'urban_model.py'
theta = [1., 0., .3/M, 100., 1.3]

@testset "Testing Potential function" begin
    val, grad = Potential.potential(xval, orig, cost_mat, theta, N, M)
    @test isapprox(val, 7974.841818981084, rtol=1e-10)
    @test isapprox(grad, [183.10355421301819, 102.37102063263552, 138.11302248813243,
    72.81557920939039, 200.87615309912633, 111.96875570837248, 138.78002577089688,
    204.86036627928468, 57.510106211976606, 242.20019302894335, 133.29465444069606,
    77.20612007724182, 74.82715790070564, 59.076373967723484, 334.34235944613624,
    239.64979695480153, 235.8373803007952, 163.09924994509691, 92.20140031513313,
    297.0120938172096, 334.3315903579696, 187.9487848703509, 165.2230124272918,
    129.08317442738394, 303.9521637836631, 210.56427458932333, 274.6283167482601,
    68.39059930808655, 255.10986061918825, 55.21971594168019, 54.59265093687789,
    135.512363971561, 269.92346245409806, 114.43050359704488, 235.59387687493626,
    204.97689400286262, 92.78467443854959, 68.84662714890538, 67.9986106624914,
    275.77024623196473, 191.07919496085438, 77.8710085030866, 188.0841587723548,
    190.17531124146512, 323.6845042354136, 135.06144376079422, 150.21427028393012,
    100.60265451861736, 245.0623673081538], rtol=1e-10)
end

@testset "Testing Hessian function" begin
    true_hess = readdlm("data/hessian_test.txt")
    hess = Potential.hessian(xval, orig, cost_mat, theta, N, M)
    @test isapprox(true_hess, hess, rtol=1e-10)
end