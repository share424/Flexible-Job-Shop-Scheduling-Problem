from fjsp import FJSP
from ga import GeneticAlgoritm

solver = GeneticAlgoritm()
fjsp = FJSP("dataset.fjs", solver)
resources = fjsp.solve(iter=10, selected_offspring=.7)
fjsp.save_as_fig('output.png', resources)
