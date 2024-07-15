def parse():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-tao_type', '--tao_type', type = str, default = 'bncg', help = 'TAO algorithm type')
	parser.add_argument('-tao_max_funcs', '--tao_max_funcs', type = int, default = 10000, help = 'TAO maximum functions evaluations')
	parser.add_argument('-tao_monitor', '--tao_monitor', action = 'store_true', help = 'TAO monitor')
	parser.add_argument('-ls', '--lagrange_s', type = float, default = 1.0, help = 'Lagrange multiplier for structural material')
	parser.add_argument('-lr', '--lagrange_r', type = float, default = 5.0, help = 'Lagrange multiplier for responsive material')
	parser.add_argument('-tao_ls_type', '--tao_ls_type', type = str, default = 'more-thuente', help = "TAO line search")
	parser.add_argument('-tao_gatol', '--tao_gatol', type = float, default = 1.0e-7, help = 'Stop if norm of gradient is less than this')
	parser.add_argument('-tao_grtol', '--tao_grtol', type = float, default = 1.0e-7, help = 'Stop if relative norm of gradient is less than this')
	parser.add_argument('-tao_gttol', '--tao_gttol', type = float, default = 1.0e-7, help = 'Stop if norm of gradient is reduced by this factor')
	parser.add_argument('-tao_view', '--tao_view', action = 'store_true', help = "View convergence details")
	parser.add_argument('-tao_max_it', '--tao_max_it', type = int, default = 100, help = 'Number of TAO iterations')
	parser.add_argument('-vs', '--volume_s', type = float, default = 0.3, help = 'Volume percentage for structural material')
	parser.add_argument('-vr', '--volume_r', type = float, default = 0.3, help = 'Volume percentage for responsive material')
	parser.add_argument('-k', '--kappa', type = float, default = 6.0e-3, help = 'Weight of Modica-Mortola')
	parser.add_argument('-e', '--epsilon', type = float, default = 2.0e-3, help = 'Phase-field regularization parameter')
	parser.add_argument('-m', '--mesh', type = str, default = 'motion.msh', help = 'Dimensions of meshed beam')
	parser.add_argument('-es', '--esmodulus', type = float, default = 0.01, help = 'Elastic Modulus for structural material')
	parser.add_argument('-er', '--ermodulus', type = float, default = 1.0, help = 'Elastic Modulus for responsive material')
	parser.add_argument('-p', '--power_p', type = float, default = 2.0, help = 'Power for elasticity interpolation')
	parser.add_argument('-s', '--steamy', type = float, default = 0.0, help = 'Initial stimulus')
	parser.add_argument('-o', '--output', type = str, default = 'beam', help = 'Output file')
	parser.add_argument('-log_view')
	options = parser.parse_args()
	return options

options = parse()

from firedrake import *
from firedrake.output import VTKFile
from petsc4py import PETSc
import time
import numpy as np

start = time.time()

# Import
mesh = Mesh(options.mesh)
Id = Identity(mesh.geometric_dimension()) #Identity tensor

# Define the function spaces
V = FunctionSpace(mesh, 'CG', 1)
VV = VectorFunctionSpace(mesh, 'CG', 1, dim = 2)

# Create initial design
rho =  Function(VV, name = "Design variable")
rho_i = Function(V, name = "Material density")
rho2 = Function(V, name = "Structural material")  # Structural material 1(Blue)
rho3 = Function(V, name = "Responsive material")  # Responsive material 2(Red)

sx = Function(V, name = "Stimulus x") # Right 
sy = Function(V, name = "Stimulus y")	# Up

# Create initial design and stimulus
###### Begin Initial Design + stimulus #####
mesh_coordinates = mesh.coordinates.dat.data[:]
M = len(mesh_coordinates)

rho2.interpolate(Constant(options.volume_s))
rho2.interpolate(Constant(1.0), mesh.measure_set("cell", 4))
rho3.interpolate(Constant(options.volume_r))
rho3.interpolate(Constant(0.0), mesh.measure_set("cell", 4))

sx.interpolate(Constant(options.steamy))
sy.interpolate(Constant(options.steamy))

# rho = as_vector([rho2, rho3])
# rho = interpolate(rho, VV)
rho = Function(VV).interpolate(as_vector([rho2, rho3]))
###### End Initial Design + stimulus #####

# Define the constant parameter used in the problem
kappa = Constant(options.kappa)
lagrange_r = Constant(options.lagrange_r)
lagrange_s = Constant(options.lagrange_s)

# Total volume of the domain |omega|
# omega = assemble(interpolate(Constant(1.0), V) * dx)
omega = assemble(Function(V).interpolate(1.0) * dx)

delta = Constant(1.0e-6)
epsilon = Constant(options.epsilon)
kappa_d_e = Constant(kappa / epsilon)
kappa_m_e = Constant(kappa * epsilon)

# Define the predescribed displacements
u_star_x = Constant((1.0, 0.0)) # Right
u_star_y = Constant((0.0, 1.0)) # Up

# Young's modulus of the beam and poisson ratio
E_v = Constant(delta)
E_s = Constant(options.esmodulus)
E_r = Constant(options.ermodulus)
ratio = options.ermodulus/options.esmodulus
nu = Constant(0.3) #nu poisson ratio

mu_v = E_v/(2 * (1 + nu))
lambda_v = (E_v * nu)/((1 + nu) * (1 - 2 * nu))

mu_s = E_s/(2 * (1 + nu))
lambda_s = (E_s * nu)/((1 + nu) * (1 - 2 * nu))

mu_r = E_r/(2 * (1 + nu))
lambda_r = (E_r * nu)/((1 + nu) * (1 - 2 * nu))

def v_v(rho):
	return 1 - rho.sub(0) - rho.sub(1)

def v_s(rho):
	return rho.sub(0)

def v_r(rho):
	return rho.sub(1)

# Define h_v(rho)=rho_v^(p)
def h_v(rho):
	return pow((1 - rho.sub(0) - rho.sub(1)), options.power_p)

# Define h_s(rho)=rho_s^(p)
def h_s(rho):
	return pow(rho.sub(0), options.power_p)

# Define h_r(rho)=rho_r^(p)
def h_r(rho):
	return pow(rho.sub(1), options.power_p)

# Define the double-well potential function
# W(x, y) = (1 - x - y)^p * (x + y)^p + (1 - x)^p * x^p + (1 - y)^p * y^p
def W(rho):
	void_func = pow((1 - rho.sub(0) - rho.sub(1)), options.power_p) * pow((rho.sub(0) + rho.sub(1)), options.power_p)
	stru_func = pow((1 - rho.sub(0)), options.power_p) * pow(rho.sub(0), options.power_p)
	resp_func = pow((1 - rho.sub(1)), options.power_p) * pow(rho.sub(1), options.power_p)
	return void_func + stru_func + resp_func

# Define strain tensor epsilon(u)
def epsilon(u):
	return 0.5 * (grad(u) + grad(u).T)

# Define the residual stresses
def sigma_A(A, Id):
	return lambda_r * tr(A) * Id + 2 * mu_r * A

# Define the stress tensor sigma_v(u) for void
def sigma_v(u, Id):
	return lambda_v * tr(epsilon(u)) * Id + 2 * mu_v * epsilon(u)

# Define the stress tensor sigma_s(u) for structural material
def sigma_s(u, Id):
	return lambda_s * tr(epsilon(u)) * Id + 2 * mu_s * epsilon(u)

# Define the stress tensor sigma_r(u) for responsive material
def sigma_r(u, Id):
	return lambda_r * tr(epsilon(u)) * Id + 2 * mu_r * epsilon(u)

# Define test function and beam displacement
vx = TestFunction(VV)
vy = TestFunction(VV)

ux = Function(VV, name = "x-displacement")
uy = Function(VV, name = "y-displacement")

px = Function(VV, name = "Adjoint variable x")
py = Function(VV, name = "Adjoint variable y")

# The left side of the beam is clamped
bcs = DirichletBC(VV, Constant((0, 0)), 7)

# Define the Modica-Mortola functional
func1 = kappa_d_e * W(rho) * dx

func2_sub1 = inner(grad(v_v(rho)), grad(v_v(rho))) * dx
func2_sub2 = inner(grad(v_s(rho)), grad(v_s(rho))) * dx
func2_sub3 = inner(grad(v_r(rho)), grad(v_r(rho))) * dx

func2 = kappa_m_e * (func2_sub1 + func2_sub2 + func2_sub3)

P = func1 + func2

# Penalty for stimulus on void + structural material
func3_x = pow(v_v(rho), 2) * pow(sx, 2) * dx
func4_x = pow(v_s(rho2), 2) * pow(sx, 2) * dx

func3_y = pow(v_v(rho), 2) * pow(sy, 2) * dx
func4_y = pow(v_s(rho2), 2) * pow(sy, 2) * dx

S = func3_x + func4_x + func3_y + func4_y

# Objective function + Modica-Mortola functional + Penalty
Obj_x = 0.5 * inner(ux - u_star_x, ux - u_star_x) * dx(4)
Obj_y = 0.5 * inner(uy - u_star_y, uy - u_star_y) * dx(4)

J = Obj_x + Obj_y + P + S

# Volume fraction penalties
func5 = lagrange_s * v_s(rho) * dx
func6 = lagrange_r * v_r(rho) * dx

# Objective function + volume penalties
JJ = J + func5 + func6

# Define the weak form for forward PDEx
a_forward_v_x = h_v(rho) * inner(sigma_v(ux, Id), epsilon(vx)) * dx
a_forward_s_x = h_s(rho) * inner(sigma_s(ux, Id), epsilon(vx)) * dx
a_forward_r_x = h_r(rho) * inner(sigma_r(ux, Id), epsilon(vx)) * dx
a_forward_x = a_forward_v_x + a_forward_s_x + a_forward_r_x

# Define the weak form for forward PDEy
a_forward_v_y = h_v(rho) * inner(sigma_v(uy, Id), epsilon(vy)) * dx
a_forward_s_y = h_s(rho) * inner(sigma_s(uy, Id), epsilon(vy)) * dx
a_forward_r_y = h_r(rho) * inner(sigma_r(uy, Id), epsilon(vy)) * dx
a_forward_y = a_forward_v_y + a_forward_s_y + a_forward_r_y

L_forward_x = inner(-u_star_x, vx) * ds(8) + sx * h_r(rho) * inner(sigma_A(Id, Id), epsilon(vx)) * dx
L_forward_y = inner(-u_star_y, vy) * ds(8) + sy * h_r(rho) * inner(sigma_A(Id, Id), epsilon(vy)) * dx

L_forward_x_s = sx * h_r(rho) * inner(sigma_A(Id, Id), epsilon(vx)) * dx
L_forward_y_s = sy * h_r(rho) * inner(sigma_A(Id, Id), epsilon(vy)) * dx

R_fwd_x = a_forward_x - L_forward_x
R_fwd_y = a_forward_y - L_forward_y

R_fwd_x_s = a_forward_x - L_forward_x_s
R_fwd_y_s = a_forward_y - L_forward_y_s

# Define the Lagrangian
a_lagrange_v_x = h_v(rho) * inner(sigma_v(ux, Id), epsilon(px)) * dx
a_lagrange_s_x = h_s(rho) * inner(sigma_s(ux, Id), epsilon(px)) * dx
a_lagrange_r_x = h_r(rho) * inner(sigma_r(ux, Id), epsilon(px)) * dx
a_lagrange_x = a_lagrange_v_x + a_lagrange_s_x + a_lagrange_r_x

a_lagrange_v_y = h_v(rho) * inner(sigma_v(uy, Id), epsilon(py)) * dx
a_lagrange_s_y = h_s(rho) * inner(sigma_s(uy, Id), epsilon(py)) * dx
a_lagrange_r_y = h_r(rho) * inner(sigma_r(uy, Id), epsilon(py)) * dx
a_lagrange_y = a_lagrange_v_y + a_lagrange_s_y + a_lagrange_r_y

L_lagrange_x = inner(-u_star_x, px) * ds(8) + sx * h_r(rho) * inner(sigma_A(Id, Id), epsilon(px)) * dx
L_lagrange_y = inner(-u_star_y, py) * ds(8) + sy * h_r(rho) * inner(sigma_A(Id, Id), epsilon(py)) * dx

R_lagrange_x = a_lagrange_x - L_lagrange_x
R_lagrange_y = a_lagrange_y - L_lagrange_y

L = JJ + R_lagrange_x + R_lagrange_y

# Define the weak form for adjoint PDE
a_adjoint_v_x = h_v(rho) * inner(sigma_v(vx, Id), epsilon(px)) * dx
a_adjoint_s_x = h_s(rho) * inner(sigma_s(vx, Id), epsilon(px)) * dx
a_adjoint_r_x = h_r(rho) * inner(sigma_r(vx, Id), epsilon(px)) * dx
a_adjoint_x = a_adjoint_v_x + a_adjoint_s_x + a_adjoint_r_x

a_adjoint_v_y = h_v(rho) * inner(sigma_v(vy, Id), epsilon(py)) * dx
a_adjoint_s_y = h_s(rho) * inner(sigma_s(vy, Id), epsilon(py)) * dx
a_adjoint_r_y = h_r(rho) * inner(sigma_r(vy, Id), epsilon(py)) * dx
a_adjoint_y = a_adjoint_v_y + a_adjoint_s_y + a_adjoint_r_y

L_adjoint_x = inner(ux - u_star_x, vx) * dx(4)
L_adjoint_y = inner(uy - u_star_y, vy) * dx(4)

R_adj_x = a_adjoint_x + L_adjoint_x
R_adj_y = a_adjoint_y + L_adjoint_y

# if ratio == 100.0:
# 	folder_file = 'secondComputationRatio100/Ratio100.pvd'
# elif ratio == 10.0:
# 	folder_file = 'secondComputationRatio10/Ratio10.pvd'
# elif ratio == 1.0:
# 	folder_file = 'secondComputationRatio1/Ratio1.pvd'
# else:
# 	folder_file = 'secondComputationRatio01/Ratio01.pvd'

folder_file = options.output

beam = VTKFile(folder_file)
dJdrho2 = Function(V, name = "Grad w.r.t rho2")
dJdrho3 = Function(V, name = "Grad w.r.t rho3")

rho_void = Function(V, name = "Void")
rho_stru = Function(V, name = "Structural")
rho_resp = Function(V, name = "Responsive")

func_Ax = Function(V, name = "Ax")
func_Ay = Function(V, name = "Ay")
func_B = Function(V, name = "B")


index_2 = [2 * i for i in range(M)]
index_3 = [2 * i + 1 for i in range(M)]

def FormObjectiveGradient(tao, x, G):

	# Print volume fraction of structural material
	volume_s = assemble(v_s(rho) * dx)/omega
	PETSc.Sys.Print("   The volume fraction(Vs) is {}".format(volume_s))

	# PETSc.Sys.Print volume fraction of responsive material
	volume_r = assemble(v_r(rho) * dx)/omega
	PETSc.Sys.Print("   The volume fraction(Vr) is {}".format(volume_r))

	# Minimization with respect to stimulus
	Ax = h_r(rho) * (lambda_r + 2 * mu_r) * tr(epsilon(px))
	B = 2 * (h_v(rho) + h_s(rho))

	Ay = h_r(rho) * (lambda_r + 2 * mu_r) * tr(epsilon(py))

	arrayB = func_B.interpolate(B).vector().array()
	arrayAx = func_Ax.interpolate(Ax).vector().array()
	arrayAy = func_Ay.interpolate(Ay).vector().array()

	arraySx = [0] * M
	arraySy = [0] * M

	for i in range(M):
		if arrayAy[i] == 0:
			arraySy[i] = 0
		elif arrayB[i] <  abs(arrayAy[i]) and arrayAy[i] > 0:
			arraySy[i] = 1
		elif arrayB[i] < abs(arrayAy[i]) and arrayAy[i] < 0:
			arraySy[i] = -1
		else:
			arraySy[i] = arrayAy[i]/arrayB[i]
	sy.vector()[:] = arraySy

	for i in range(M):
		if arrayAx[i] == 0:
			arraySx[i] = 0
		elif arrayB[i] <  abs(arrayAx[i]) and arrayAx[i] > 0:
			arraySx[i] = 1
		elif arrayB[i] < abs(arrayAx[i]) and arrayAx[i] < 0:
			arraySx[i] = 1
		else:
			arraySx[i] = arrayAx[i]/arrayB[i]
	sx.vector()[:] = arraySx

	i = tao.getIterationNumber()
	if (i%5) == 0:
		rho_i.interpolate(rho.sub(1) - rho.sub(0))
		rho_stru.interpolate(rho.sub(0))
		rho_resp.interpolate(rho.sub(1))
		rho_void.interpolate(1 - rho.sub(0) - rho.sub(1))
		solve(R_fwd_x_s == 0, ux, bcs = bcs)
		solve(R_fwd_y_s == 0, uy, bcs = bcs)
		beam.write(rho_i, sx, sy, rho_stru, rho_resp, rho_void, ux, uy, time = i)

	with rho.dat.vec as rho_vec:
		rho_vec.set(0.0)
		rho_vec.axpy(1.0, x)

	# Solve forward PDEs
	solve(R_fwd_x == 0, ux, bcs = bcs)
	solve(R_fwd_y == 0, uy, bcs = bcs)

	# Solve adjoint PDEs
	solve(R_adj_x == 0, px, bcs = bcs)
	solve(R_adj_y == 0, py, bcs = bcs)

	# Evaluate the objective function
	objective_value = assemble(J)
	PETSc.Sys.Print("   The value of objective function is {}".format(objective_value))

	# Compute gradiet w.r.t rho2 and rho3 and s
	dJdrho2.interpolate(assemble(derivative(L, rho.sub(0))).riesz_representation(riesz_map="l2"))
	dJdrho2.interpolate(Constant(0.0), mesh.measure_set("cell", 4))

	dJdrho3.interpolate(assemble(derivative(L, rho.sub(1))).riesz_representation(riesz_map="l2"))
	dJdrho3.interpolate(Constant(0.0), mesh.measure_set("cell", 4))

	G.setValues(index_2, dJdrho2.vector().array())
	G.setValues(index_3, dJdrho3.vector().array())

	f_val = assemble(L)
	return f_val

# Setting lower and upper bounds
# lb = as_vector((0, 0))
# ub = as_vector((1, 1))
# lb = interpolate(lb, VV)
# ub = interpolate(ub, VV)
lb = Function(VV).interpolate(as_vector((0, 0)))
ub = Function(VV).interpolate(as_vector((1, 1)))

with lb.dat.vec as lb_vec:
	rho_lb = lb_vec

with ub.dat.vec as ub_vec:
	rho_ub = ub_vec

# Setting TAO solver
tao = PETSc.TAO().create(PETSc.COMM_WORLD)
tao.setType('bncg')
tao.setObjectiveGradient(FormObjectiveGradient, None)
tao.setVariableBounds(rho_lb, rho_ub)
tao.setFromOptions()

# Initial design guess
with rho.dat.vec as rho_vec:
	x = rho_vec.copy()

# Solve the optimization problem
tao.solve(x)
tao.destroy()

end = time.time()
PETSc.Sys.Print("\nExecution time (in seconds):", (end - start))