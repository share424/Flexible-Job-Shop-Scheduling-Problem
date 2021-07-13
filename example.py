from fjsp import FJSP, save_as_excel, save_as_fig, save_as_excel
from ga import GeneticAlgorithm

solver = GeneticAlgorithm()
fjsp = FJSP("dataset.fjs", solver)
resources = fjsp.solve(iter=1, selected_offspring=.8)
save_as_fig('output.png', resources)
save_as_excel('output.xlsx', resources)
