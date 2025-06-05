import numpy as np
import matplotlib.pyplot as plt
import fenics as fn
from GwFlow import GwFlowSolver
from RandomProcess import SquaredExponential


class Model:
    def __init__(self, resolution, field_mean, field_stdev, mkl, lamb):
        """
        This class is basically a wrapper
        around GwFlowSolver and RandomProcess.
        It has some functions that makes it
        simple to access important features.
        """
        
        # Set up all the parameters needed
        # for the solver and the ransom field.
        self.resolution = resolution
        self.field_mean = field_mean
        self.field_stdev = field_stdev
        self.mkl = mkl
        self.lamb = lamb
        self.parameters = None
        
        # Initialise a solver.
        self.solver = GwFlowSolver(self.resolution,
                                   self.field_mean,
                                   self.field_stdev)
        _dof_coords = self.solver.V.tabulate_dof_coordinates().reshape((-1, 2))
        _dof_indices = self.solver.V.dofmap().dofs()
        self.coords = _dof_coords[_dof_indices, :] 
        
        # initialise a random process given the solver mesh.
        self.random_process = SquaredExponential(self.coords,
                                                 self.mkl,
                                                 self.lamb)
        
        # Compute the eigenpairs of the covariance
        # matrix in the random process.
        self.random_process.compute_eigenpairs()
        
    def solve(self, parameters=None):
        """
        Solve the problem, given a vector of modes.
        """
        self.random_process.generate(parameters)
        self.parameters = self.random_process.parameters
        self.solver.set_conductivity(self.random_process.random_field)
        self.solver.solve()
        
    def get_data(self, datapoints):
        """
        Get data from a list of coordinates.
        """
        return self.solver.get_data(datapoints)
        
    def get_outflow(self):
        return self.solver.get_outflow()
        
    def plot(self, limits=[0, 0], lognormal=True):
        """
        This method plots both the random firld and the solution.
        """
        
        # First, contruct a random field, given the field parameters.
        if lognormal:
            random_field = self.field_mean + \
                           self.field_stdev*self.random_process.random_field
        else:
            random_field = np.exp(self.field_mean +
                                  self.field_stdev*self.random_process.random_field)
        
        # Set contour levels.
        if any(limits):
            contour_levels_field = \
                np.linspace(limits[0], limits[1], 100)
        else:
            contour_levels_field = \
                np.linspace(min(random_field), max(random_field), 100)
        
        # Then extract the solution from every node in the solver.
        solution = self.solver.h.vector()[:]
            
        # Set the contour levels.
        contour_levels_solution = np.linspace(min(solution),
                                              max(solution),
                                              100)
        
        # Plot field and solution.
        fig, axes = plt.subplots(nrows=1, ncols=2,
                                 figsize=(24, 9))
        axes[0].set_title('Transmissivity Field',
                          fontdict={'fontsize': 24})
        axes[0].tick_params(labelsize=16)
        field = axes[0].tricontourf(self.coords[:, 0],
                                    self.coords[:, 1],
                                    random_field, 
                                    levels=contour_levels_field,
                                    cmap='plasma')
        fig.colorbar(field, ax=axes[0])
        
        axes[1].set_title('Solution', fontdict={'fontsize': 24})
        axes[1].tick_params(labelsize=16)
        solution = axes[1].tricontourf(self.coords[:, 0],
                                       self.coords[:, 1],
                                       solution, 
                                       levels=contour_levels_solution,
                                       cmap='plasma')
        fig.colorbar(solution, ax=axes[1])


def model_wrapper(my_model, theta, datapoints):
    """
    This solves a groundwater model given theta and returns the solution
    on the specified datapoints. Argument my_model is an object
    of class Model defined in mlda/model.py.
    """
    my_model.solve(theta)
    return my_model.get_data(datapoints)


def project_eigenpairs(model_fine, model_coarse):
    """
    Projects eigenpairs from a fine model to a coarse model.
    """
    model_coarse.random_process.eigenvalues[:] = \
        model_fine.random_process.eigenvalues
    for i in range(model_coarse.mkl):
        psi_fine = fn.Function(model_fine.solver.V)
        psi_fine.vector()[:] = \
            model_fine.random_process.eigenvectors[:, i]
        psi_coarse = fn.project(psi_fine, model_coarse.solver.V)
        model_coarse.random_process.eigenvectors[:, i] = \
            psi_coarse.vector()[:]

# import torch
# import matplotlib.pyplot as plt
# import fenics as fn
# from GwFlow import GwFlowSolver
# from RandomProcess import SquaredExponential  # torch-based version


# class Model:
#     def __init__(self, resolution, field_mean, field_stdev, mkl, lamb):
#         """
#         Wrapper around GwFlowSolver and RandomProcess using PyTorch.
#         """
#         self.resolution = resolution
#         self.field_mean = field_mean
#         self.field_stdev = field_stdev
#         self.mkl = mkl
#         self.lamb = lamb
#         self.parameters = None

#         self.solver = GwFlowSolver(resolution, field_mean, field_stdev)

#         # Extract DoF coordinates from FEniCS
#         _dof_coords = self.solver.V.tabulate_dof_coordinates().reshape((-1, 2))
#         _dof_indices = self.solver.V.dofmap().dofs()
#         self.coords = torch.tensor(_dof_coords[_dof_indices, :], dtype=torch.float32)

#         # Initialize random process
#         self.random_process = SquaredExponential(self.coords, mkl, lamb)
#         self.random_process.compute_eigenpairs()

#     def solve(self, parameters=None):
#         self.random_process.generate(parameters)
#         self.parameters = self.random_process.parameters
#         # FEniCS expects a NumPy array
#         conductivity = self.random_process.random_field.detach().numpy()
#         self.solver.set_conductivity(conductivity)
#         self.solver.solve()

#     def get_data(self, datapoints):
#         return self.solver.get_data(datapoints)

#     def get_outflow(self):
#         return self.solver.get_outflow()

#     def plot(self, limits=[0, 0], lognormal=True):
#         if lognormal:
#             random_field = self.field_mean + self.field_stdev * self.random_process.random_field
#         else:
#             random_field = torch.exp(self.field_mean + self.field_stdev * self.random_process.random_field)

#         field_np = random_field.detach().numpy()
#         coords_np = self.coords.detach().numpy()

#         # Contour levels
#         contour_levels_field = (
#             torch.linspace(*limits, 100).numpy() if any(limits)
#             else torch.linspace(field_np.min(), field_np.max(), 100)
#         )

#         solution = self.solver.h.vector()[:]
#         contour_levels_solution = torch.linspace(solution.min(), solution.max(), 100)

#         fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 9))

#         axes[0].set_title("Transmissivity Field", fontdict={'fontsize': 24})
#         axes[0].tick_params(labelsize=16)
#         f = axes[0].tricontourf(coords_np[:, 0], coords_np[:, 1],
#                                 field_np, levels=contour_levels_field, cmap='plasma')
#         fig.colorbar(f, ax=axes[0])

#         axes[1].set_title("Solution", fontdict={'fontsize': 24})
#         axes[1].tick_params(labelsize=16)
#         s = axes[1].tricontourf(coords_np[:, 0], coords_np[:, 1],
#                                 solution, levels=contour_levels_solution, cmap='plasma')
#         fig.colorbar(s, ax=axes[1])

# def model_wrapper(my_model, theta, datapoints):
#     theta_tensor = torch.tensor(theta, dtype=torch.float32)
#     my_model.solve(theta_tensor)
#     return my_model.get_data(datapoints)


# def project_eigenpairs(model_fine, model_coarse):
#     model_coarse.random_process.eigenvalues[:] = model_fine.random_process.eigenvalues.clone()
#     for i in range(model_coarse.mkl):
#         psi_fine = fn.Function(model_fine.solver.V)
#         psi_fine.vector()[:] = model_fine.random_process.eigenvectors[:, i].detach().numpy()
#         psi_coarse = fn.project(psi_fine, model_coarse.solver.V)
#         model_coarse.random_process.eigenvectors[:, i] = \
#             torch.tensor(psi_coarse.vector()[:], dtype=torch.float32)
