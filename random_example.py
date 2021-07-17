from fjsp import FJSP, save_as_fig, save_as_excel
from ga import GeneticAlgorithm

solver = GeneticAlgorithm()
fjsp = FJSP("dataset.fjs", solver)
solver.problem = fjsp.problem
chromosome = solver.random_selection()
resources = solver.decode_chromosome(chromosome)
save_as_fig('random_output.png', resources)
save_as_excel('random_output.xlsx', resources)
