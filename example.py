from fjsp import FJSP, save_as_fig
from ga import GeneticAlgoritm

solver = GeneticAlgoritm()
fjsp = FJSP("dataset.fjs", solver)
resources = fjsp.solve(iter=10, selected_offspring=.7)
save_as_fig('output.png', resources)
