def parse():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-tao_type', '--tao_type', type = str, default = 'bncg', help = 'TAO algorithm type')
	parser.add_argument('-tao_max_funcs', '--tao_max_funcs', type = int, default = 10000, help = 'TAO maximum functions evaluations')
	parser.add_argument('-tao_monitor', '--tao_monitor', action = 'store_true', help = 'TAO monitor')
	parser.add_argument('-ls', '--lagrange_s', type = float, default = 1.0, help = 'Lagrange multiplier for structural material')
	parser.add_argument('-lr', '--lagrange_r', type = float, default = 5.0, help = 'Lagrange multiplier for responsive material')
	parser.add_argument('-tao_ls_type', '--tao_ls_type', type = str, default = 'more-thuente', help = "TAO line search")
	parser.add_argument('-tao_view', '--tao_view', action = 'store_true', help = "View convergence details")
	parser.add_argument('-tao_max_it', '--tao_max_it', type = int, default = 100, help = 'Number of TAO iterations')
	parser.add_argument('-tao_gatol', '--tao_gatol', type = float, default = 1.0e-7, help = 'Stop if norm of gradient is less than this')
	parser.add_argument('-tao_grtol', '--tao_grtol', type = float, default = 1.0e-7, help = 'Stop if relative norm of gradient is less than this')
	parser.add_argument('-tao_gttol', '--tao_gttol', type = float, default = 1.0e-7, help = 'Stop if norm of gradient is reduced by this factor')
	parser.add_argument('-vs', '--volume_s', type = float, default = 0.3, help = 'Target volume for structural material')
	parser.add_argument('-vr', '--volume_r', type = float, default = 0.3, help = 'Target volume for responsive material')
	parser.add_argument('-k', '--kappa', type = float, default = 6.0e-3, help = 'Weight of perimeter penalization')
	parser.add_argument('-e', '--epsilon', type = float, default = 2.0e-3, help = 'Phase-field regularization parameter')
	parser.add_argument('-m', '--mesh', type = str, default = 'motion.msh', help = 'Design domain mesh')
	parser.add_argument('-es', '--esmodulus', type = float, default = 0.01, help = 'Elastic Modulus for structural material')
	parser.add_argument('-er', '--ermodulus', type = float, default = 1.0, help = 'Elastic Modulus for responsive material')
	parser.add_argument('-p', '--power_p', type = float, default = 2.0, help = 'Power for elasticity interpolation + triple well potential')
	parser.add_argument('-s', '--steamy', type = float, default = 1.0, help = 'Initial stimulus')
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

# Import gmesh
mesh = Mesh(options.mesh)
Id = Identity(mesh.geometric_dimension()) #Identity tensor

# Define the function spaces
V = FunctionSpace(mesh, 'CG', 1)
VV = VectorFunctionSpace(mesh, 'CG', 1, dim = 2)
VVV = VectorFunctionSpace(mesh, 'CG', 1, dim = 3)

# Create initial design
###### Begin Initial Design #####
mesh_coordinates = mesh.coordinates.dat.data[:]
M = len(mesh_coordinates)

rho =  Function(VVV, name = "Design variable")
rho_i = Function(V, name = "Material density")
rho2 = Function(V, name = "Structural material")  # Structural material 1(Blue)
rho3 = Function(V, name = "Responsive material")  # Responsive material 2(Red)
s = Function(V, name = "Stimulus")

rho2.interpolate(Constant(options.volume_s))
rho2.interpolate(Constant(1.0), mesh.measure_set("cell", 4))
rho3.interpolate(Constant(options.volume_r))
rho3.interpolate(Constant(0.0), mesh.measure_set("cell", 4))
s.interpolate(Constant(options.steamy))

# rho = as_vector([rho2, rho3, s])
# rho = interpolate(rho, VVV)
rho = Function(VVV).interpolate(as_vector([rho2, rho3, s]))
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

# Define predescribed displacement
u_star = Constant((0, 1.0))

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

# Define h_v(rho)=rho_v^p, p = 2
def h_v(rho):
	return pow((1 - rho.sub(0) - rho.sub(1)), options.power_p)

# Define h_s(rho)=rho_s^(p)
def h_s(rho):
	return pow(rho.sub(0), options.power_p)

# Define h_r(rho)=rho_r^(p)
def h_r(rho):
	return pow(rho.sub(1), options.power_p)

def s_s(rho):
	return rho.sub(2)

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
	return lambda_v * div(u) * Id + 2 * mu_v * epsilon(u)

# Define the stress tensor sigma_s(u) for structural material
def sigma_s(u, Id):
	return lambda_s * div(u) * Id + 2 * mu_s * epsilon(u)

# Define the stress tensor sigma_r(u) for responsive material
def sigma_r(u, Id):
	return lambda_r * div(u) * Id + 2 * mu_r * epsilon(u)


# Define test function and beam displacement
v = TestFunction(VV)
u = Function(VV, name = "Displacement")
us = Function(VV, name = "Displacement")
p = Function(VV, name = "Adjoint variable")

# The left side of the beam is clamped
bcs = DirichletBC(VV, Constant((0, 0)), 7)

# Define the objective function
J = 0.5 * inner(u - u_star, u - u_star) * dx(4)
func1 = kappa_d_e * W(rho) * dx

func2_sub1 = inner(grad(v_v(rho)), grad(v_v(rho))) * dx
func2_sub2 = inner(grad(v_s(rho)), grad(v_s(rho))) * dx
func2_sub3 = inner(grad(v_r(rho)), grad(v_r(rho))) * dx

func2 = kappa_m_e * (func2_sub1 + func2_sub2 + func2_sub3)

func3 = lagrange_s * v_s(rho) * dx
func4 = lagrange_r * v_r(rho) * dx

# stimulus penalty
func5 = pow(v_v(rho), 2) * pow(s_s(rho), 2) * dx
func6 = pow(v_s(rho), 2) * pow(s_s(rho), 2) * dx

# Objective function + Modica-Mortola functional
P = func1 + func2 + func3 + func4 + func5 + func6
JJ = J + P

# Define the weak form of the state PDE
a_forward_v = h_v(rho) * inner(sigma_v(u, Id), epsilon(v)) * dx
a_forward_s = h_s(rho) * inner(sigma_s(u, Id), epsilon(v)) * dx
a_forward_r = h_r(rho) * inner(sigma_r(u, Id), epsilon(v)) * dx
a_forward = a_forward_v + a_forward_s + a_forward_r

L_forward = inner(-u_star, v) * ds(8) + s_s(rho) * h_r(rho) * inner(sigma_A(Id, Id), epsilon(v)) * dx
L_forward_s = s_s(rho) * h_r(rho) * inner(sigma_A(Id, Id), epsilon(v)) * dx
R_fwd = a_forward - L_forward
R_fwd_s = a_forward - L_forward_s

# Define the Lagrangian
a_lagrange_v = h_v(rho) * inner(sigma_v(u, Id), epsilon(p)) * dx
a_lagrange_s = h_s(rho) * inner(sigma_s(u, Id), epsilon(p)) * dx
a_lagrange_r = h_r(rho) * inner(sigma_r(u, Id), epsilon(p)) * dx
a_lagrange   = a_lagrange_v + a_lagrange_s + a_lagrange_r

L_lagrange = inner(-u_star, p) * ds(8) + s_s(rho) * h_r(rho) * inner(sigma_A(Id, Id), epsilon(p)) * dx
R_lagrange = a_lagrange - L_lagrange
L = JJ - R_lagrange

# Define the weak form of the adjoint PDE
a_adjoint_v = h_v(rho) * inner(sigma_v(v, Id), epsilon(p)) * dx
a_adjoint_s = h_s(rho) * inner(sigma_s(v, Id), epsilon(p)) * dx
a_adjoint_r = h_r(rho) * inner(sigma_r(v, Id), epsilon(p)) * dx
a_adjoint = a_adjoint_v + a_adjoint_s + a_adjoint_r

L_adjoint = inner(u - u_star, v) * dx(4)
R_adj = a_adjoint - L_adjoint

# if ratio == 100.0:
# 	folder_file = 'directFirstComputationRatio100/Ratio100.pvd'
# elif ratio == 10.0:
# 	folder_file = 'directFirstComputationRatio10/Ratio10.pvd'
# elif ratio == 1.0:
# 	folder_file = 'directFirstComputationRatio1/Ratio1.pvd'
# else:
# 	folder_file = 'directFirstComputationRatio01/Ratio01.pvd'
folder_file = options.output

beam = VTKFile(folder_file)
dJdrho2 = Function(V, name = "Grad w.r.t rho2")
dJdrho3 = Function(V, name = "Grad w.r.t rho3")
rho_void = Function(V, name = "Void")
rho_resp = Function(V, name = "Responsive")
rho_stru = Function(V, name = "Structural")

dJds = Function(V)
stimulus = Function(V, name = "Stimulus")

index_2 = [3 * i for i in range(M)]
index_3 = [3 * i + 1 for i in range(M)]
index_s = [3 * i + 2 for i in range(M)]

def FormObjectiveGradient(tao, x, G):

	# Print volume fraction of structural material
	volume_s = assemble(v_s(rho) * dx)/omega
	PETSc.Sys.Print("   The volume fraction(Vs) is {}".format(volume_s))

	# Print volume fraction of responsive material
	volume_r = assemble(v_r(rho) * dx)/omega
	PETSc.Sys.Print("   The volume fraction(Vr) is {}".format(volume_r))

	i = tao.getIterationNumber()
	if (i%5) == 0:
		rho_i.interpolate(rho.sub(1) - rho.sub(0))
		stimulus.interpolate(rho.sub(2))
		rho_stru.interpolate(rho.sub(0))
		rho_void.interpolate(1 - rho.sub(0) - rho.sub(1))
		rho_resp.interpolate(rho.sub(1))

		solve(R_fwd_s == 0, u, bcs = bcs)
		beam.write(rho_i, stimulus, rho_stru, rho_resp, rho_void, u, time = i)

	with rho.dat.vec as rho_vec:
		rho_vec.set(0.0)
		rho_vec.axpy(1.0, x)


	# Solve forward PDE
	solve(R_fwd == 0, u, bcs = bcs)

	# Solve adjoint PDE
	solve(R_adj == 0, p, bcs = bcs)

	# Evaluate the objective function
	objective_value = assemble(J)
	PETSc.Sys.Print("   The value of objective function is {}".format(objective_value))

	# Compute gradiet w.r.t rho2 and rho3
	dJdrho2.interpolate(assemble(derivative(L, rho.sub(0))).riesz_representation(riesz_map="l2"))
	dJdrho2.interpolate(Constant(0.0), mesh.measure_set("cell", 4))
	
	dJdrho3.interpolate(assemble(derivative(L, rho.sub(1))).riesz_representation(riesz_map="l2"))
	dJdrho3.interpolate(Constant(0.0), mesh.measure_set("cell", 4))
	dJds.interpolate(assemble(derivative(L, rho.sub(2))).riesz_representation(riesz_map="l2"))

	G.setValues(index_2, dJdrho2.vector().array())
	G.setValues(index_3, dJdrho3.vector().array())
	G.setValues(index_s, dJds.vector().array())

	f_val = assemble(L)
	return f_val

# Setting lower and upper bounds
# lb = as_vector((0, 0, -1))
# ub = as_vector((1, 1, 1))
# lb = interpolate(lb, VVV)
# ub = interpolate(ub, VVV)
lb = Function(VVV).interpolate(as_vector((0, 0, -1)))
ub = Function(VVV).interpolate(as_vector((1, 1, 1)))

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
