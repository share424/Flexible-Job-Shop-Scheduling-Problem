from ga import GAFJSP, TwoPointCrossover, UniformCrossover, POXCrossover
from fjsp import FJSP, Problem, Job, Operation, Machine

ga = GAFJSP()

job_1 = Job(1)
operation_11 = Operation(1, job_1.id)
operation_11.add_machine(Machine(1, 2))
operation_11.add_machine(Machine(2, 6))
operation_11.add_machine(Machine(3, 5))
operation_11.add_machine(Machine(4, 3))
operation_11.add_machine(Machine(5, 4))
job_1.add_operation(operation_11)

operation_12 = Operation(2, job_1.id)
operation_12.add_machine(Machine(2, 8))
operation_12.add_machine(Machine(4, 4))
job_1.add_operation(operation_12)

job_2 = Job(2)
operation_21 = Operation(1, job_2.id)
operation_21.add_machine(Machine(1, 3))
operation_21.add_machine(Machine(3, 6))
operation_21.add_machine(Machine(5, 5))
job_2.add_operation(operation_21)

operation_22 = Operation(2, job_2.id)
operation_22.add_machine(Machine(1, 4))
operation_22.add_machine(Machine(2, 6))
operation_22.add_machine(Machine(3, 5))
job_2.add_operation(operation_22)

operation_23 = Operation(3, job_2.id)
operation_23.add_machine(Machine(2, 7))
operation_23.add_machine(Machine(3, 11))
operation_23.add_machine(Machine(4, 5))
operation_23.add_machine(Machine(5, 8))
job_2.add_operation(operation_23)

problem = Problem([job_1, job_2], 5)
ga.solve(problem)

# p1 = [2, 2, 1, 3, 2, 4, 5, 3, 1, 5, 4, 3, 1, 2]
# p2 = [5, 1, 2, 1, 4, 3, 2, 5, 3, 2, 2, 1, 4, 3]

# co = POXCrossover()
# print(co.crossover(p1, p2))

