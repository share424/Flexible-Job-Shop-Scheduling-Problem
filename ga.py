from fjsp import Solver, Problem, Job, Operation, Machine, Task, Resource
from typing import List, Tuple
import numpy as np

class Chromosome:
    def __init__(self, machine_selection: List[int], operation_sequence: List[int]):
        self.machine_selection = machine_selection
        self.operation_sequence = operation_sequence
        self.fitness = 0

    def set_fitness(self, fitness):
        self.fitness = fitness

    def __str__(self) -> str:
        return "[" + (", ".join(str(i) for i in (self.machine_selection + self.operation_sequence))) + "]\n"
        
    def __repr__(self):
        return str(self)

class ParentSelector:
    def __init__(self, name):
        self.name = name

    def select_parent(self, population: List[Chromosome]) -> Chromosome:
        pass

class RoulleteWheel(ParentSelector):
    def __init__(self):
        super(RoulleteWheel, self).__init__("roullete_wheel")

    def select_parent(self, population: List[Chromosome]):
        total_fitness = np.sum([c.fitness for c in population])
        p = np.random.rand()
        pointer = 0
        for i, c in enumerate(population):
            r = c/total_fitness
            pointer += r
            if p < pointer:
                return i
        return len(population) - 1

class Tournament(ParentSelector):
    def __init__(self):
        super(Tournament, self).__init__("tournament")
    
    def select_parent(self, population: List[Chromosome], k=3):
        candidates = np.random.choice(len(population), size=3, replace=False)
        best_candidate = candidates[0]
        for i in candidates:
            if population[i].fitness > population[best_candidate].fitness:
                best_candidate = i
        return i
        
class Crossover:
    def __init__(self, name: str):
        self.name = name

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[Chromosome, Chromosome]:
        pass

class TwoPointCrossover(Crossover):
    def __init__(self):
        super(TwoPointCrossover, self).__init__("two_point")
    
    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        points = np.random.choice(len(p1), size=2, replace=False)
        points.sort()
        ms1 = p1[:points[0]] + p2[points[0]:points[1]] + p1[points[1]:]
        ms2 = p2[:points[0]] + p1[points[0]:points[1]] + p2[points[1]:]
        return ms1, ms2

class UniformCrossover(Crossover):
    def __init__(self):
        super(UniformCrossover, self).__init__("uniform")

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        swap = np.random.randint(2, size=len(p1))

        ms1 = np.copy(p1)
        ms2 = np.copy(p2)

        for i, j in enumerate(swap):
            if j == 1:
                temp = ms1[i]
                ms1[i] = ms2[i]
                ms1[i] = temp

        return ms1, ms2

class POXCrossover(Crossover):
    def __init__(self):
        super(POXCrossover, self).__init__("pox")

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        # generate sub-joint between p1 and p2
        sub_joints = set()
        for i in p1:
            if i in p2:
                sub_joints.add(i)
        for i in p2:
            if i in p1:
                sub_joints.add(i)

        sub_joints = list(sub_joints)

        # divide into 2 parts
        np.random.shuffle(sub_joints)
        js1 = sub_joints[:len(sub_joints)//2]
        js2 = sub_joints[len(sub_joints)//2:]

        c1, c2 = [], []

        j = 0
        for i in p2:
            if i not in js1:
                c1.append(i)
                j += 1
            
            while j < len(p1) and p1[j] in js1:
                c1.append(p1[j])
                j += 1
        
        j = 0
        for i in p1:
            if i not in js2:
                c2.append(i)
                j += 1
            
            while j < len(p2) and p2[j] in js2:
                c2.append(p2[j])
                j += 1
            
        return c1, c2

class Mutator:
    def __init__(self, name, p: float):
        self.name = name
        self.p = p

    def mutate(self, p: List[int]) -> List[int]:
        pass

class MSMutator(Mutator):
    def __init__(self, p: float):
        super(MSMutator, self).__init__("ms", p)

    def mutate(self, p: List[int], problem: Problem) -> List[int]:
        pc = np.copy(p)
        for i, m in enumerate(p):
            prob = np.random.rand()
            if prob < self.p:
                opr = problem.get_operation_by_index(i)
                pc[i] = opr.get_lowest_machine() + 1
        return pc

class OSMutator(Mutator):
    def __init__(self, p: float):
        super(OSMutator, self).__init__("os", p)
    
    def mutate(self, p: List[int]) -> List[int]:
        prob = np.random.rand()
        pc = np.copy(p)
        if prob < self.p:
            np.random.shuffle(pc)
        return pc

class GeneticAlgorithm(Solver):
    def __init__(self):
        super(GeneticAlgorithm, self).__init__("Genetic Algorithm")
        self.parent_selectors: List[ParentSelector] = [
            RoulleteWheel(),
            Tournament()
        ]
        self.crossovers: List[Crossover] = [
            TwoPointCrossover(),
            UniformCrossover(),
            POXCrossover()
        ]
    
    def get_parent_selector(self, selector) -> ParentSelector:
        for p in self.parent_selectors:
            if p.name == selector:
                return p
        raise ValueError(f"Parent selector {selector} is not defined")

    def get_crossover(self, crossover) -> Crossover:
        for c in self.crossovers:
            if c.name == crossover:
                return c
        raise ValueError(f"Crossover {crossover} is not defined")
    
    def global_selection(self) -> Chromosome:
        machine_selection = []
        operation_sequence = []

        # 1. Create a new array to record all machines’ processing time, initialize each element to 0;
        time_array = [0 for _ in range(self.problem.n_machine)]

        # 2. Select a job randomly and insure one job to be selected only once, then select the first operation of the job;
        ms_temp = [[] for _ in range(len(self.problem.jobs))]
        for job in self.problem.get_shuffled_job():
            ms: List[int] = []
            for operation in job.operations:
                # 3. Add the processing time of each machine in the available machines and the corresponding 
                #    machine’s time in the time array together
                added_time = []
                for machine in operation.machines:
                    added_time.append(time_array[machine.index] + machine.operation_time)
                
                # 4. Compare the added time to find the shortest time, then select the index k of the machine which has the shortest
                #    time. If there is the same time among different machines, a machine is selected randomly among them;
                k = np.argmin(added_time)

                # 5. Set the allele which corresponds to the current operation in the MS part to k;
                ms.append(k + 1)

                # 6. Add the current selected machine’s processing time and its corresponding allele in the 
                #    time array together in order to update the time array;
                selected_machine = operation.machines[k]
                time_array[selected_machine.index] += selected_machine.operation_time

                # 7. Select the next operation of the current job, and execute
                #    Step 3 to Step 6 until all operations of the current job are
                #    selected, then go to Step 8;

                # 8. Go to step 2 until all jobs are all selected once

                # set the operation sequence allele
                operation_sequence.append(job.id)
            ms_temp[job.index] = ms
        for ms in ms_temp:
            for i in ms:
                machine_selection.append(i)

        np.random.shuffle(operation_sequence)
        return Chromosome(machine_selection, operation_sequence)

    def local_selection(self):
        machine_selection = []
        operation_sequence = []

        # 1. In order to record all machines’ processing time, create a
        #    new array (called time array), the length equals to L, and
        #    set each element 0;
        

        # 2. Select the first job, and its first operation;
        ms_temp = [[] for _ in range(len(self.problem.jobs))]
        for job in self.problem.get_shuffled_job():
            time_array = [0 for _ in range(self.problem.L)]
            ms: List[int] = []
            for operation in job.operations:
                # 3. Set each allele 0 in the array;
                #    skip

                # 4. Add the processing time of each machine in the alternative
                #    machine set and the corresponding machines’ time
                #    in the array together;
                added_time = []
                for machine in operation.machines:
                    added_time.append(time_array[machine.index] + machine.operation_time)
                
                # 5. Compare the added time to find the shortest time, then select the index k of the machine which has the shortest
                #    time. If there is the same time among different machines, a machine is selected randomly among them;
                k = np.argmin(added_time)

                # 6. Set the allele which corresponds to the current operation in the MS part to k;
                ms.append(k + 1)

                # 7. Add the current selected machine’s processing time and its corresponding allele in the 
                #    time array together in order to update the time array;
                selected_machine = operation.machines[k]
                time_array[selected_machine.index] += selected_machine.operation_time

                # 8. Select the next operation of the current job, and go to
                #    Step 4 until all the operations of the current job are
                #    selected, then go to Step 9;

                # 9. Select the next job, and select the first operation of the current job;

                # 10. Go to Step 3 until all jobs are selected once
                # set the operation sequence allele
                operation_sequence.append(job.id)
            ms_temp[job.index] = ms
        for ms in ms_temp:
            for i in ms:
                machine_selection.append(i)

        np.random.shuffle(operation_sequence)
        return Chromosome(machine_selection, operation_sequence)

    def random_selection(self):
        machine_selection = []
        operation_sequence = []

        for job in self.problem.jobs:
            for operation in job.operations:
                selected_machine_idx = operation.get_random_machine(return_index=True)
                machine_selection.append(selected_machine_idx + 1)
                operation_sequence.append(job.id)

        np.random.shuffle(operation_sequence)
        return Chromosome(machine_selection, operation_sequence)

    def init_population(self, population_amount, gs, ls, rs):
        assert gs + ls + rs != 1, "The initialization population fragment sum is not 1"

        self.population: List[Chromosome] = []

        for _ in range(int(gs * population_amount)):
            chromosome = self.global_selection()
            self.population.append(chromosome)
        
        for _ in range(int(ls * population_amount)):
            chromosome = self.local_selection()
            self.population.append(chromosome)

        for _ in range(int(rs * population_amount)):
            chromosome = self.random_selection()
            self.population.append(chromosome)

    def is_valid_chromosome(self, chromosome: Chromosome) -> bool:
        for i, m in enumerate(chromosome.machine_selection):
            opr = self.problem.get_operation_by_index(i)
            if opr.get_machine_by_id(m) == None:
                return False
        return True

    def fix_chromosome(self, chromosome: Chromosome) -> Chromosome:
        for i, m in enumerate(chromosome.machine_selection):
            opr = self.problem.get_operation_by_index(i)
            chromosome.machine_selection[i] = np.min([m, len(opr.machines)])
        return chromosome
    
    def decode_chromosome(self, chromosome: Chromosome):
        # 1. Convert machine selection to machine matrix and time matrix
        machine_matrix = []
        time_matrix = []
        i = 0
        for job in self.problem.jobs:
            used_machine = []
            used_time = []
            for operation in job.operations:
                machine_idx = chromosome.machine_selection[i]
                used_machine.append(operation.machines[machine_idx - 1].id)
                used_time.append(operation.machines[machine_idx - 1].operation_time)
                i += 1
            machine_matrix.append(used_machine)
            time_matrix.append(used_time)
        
        # 2. Decode operation sequence
        resources: List[Resource] = [Resource(i + 1) for i in range(self.problem.n_machine)]
        # variable to track current operation on job-n. Default is 1st operation
        current_job_operations = [1 for _ in range(len(self.problem.jobs))]
        for job_id in chromosome.operation_sequence:
            operation_id = current_job_operations[job_id - 1]
            job = self.problem.get_job_by_id(job_id)
            if job == None:
                raise ValueError(f"Job with id {job_id} is not found")
            operation = job.get_operation_by_id(operation_id)
            if operation == None:
                raise ValueError(f"Operation with id {operation_id} is not found")
            selected_machine_id = machine_matrix[job.index][operation.index]
            selected_machine = operation.get_machine_by_id(selected_machine_id)
            
            resource = resources[selected_machine.index]
            # find all idle time
            idle_times = resource.find_idle_time()
            
            # let's check if the operation can fit in the idle time
            # 1. select idle time that the start_time is >= last operation
            last_operation = job.get_operation_by_id(operation_id - 1)
            
            last_operation_time = 0
            if last_operation != None:
                # there is last operation, it means this operation need to be inserted after the last operation
                last_operation_machine = machine_matrix[job.index][last_operation.index]
                last_machine = last_operation.get_machine_by_id(last_operation_machine)
                last_resource = resources[last_machine.index]
                last_task = last_resource.find_operation(job_id, last_operation.id)
                if last_task != None:
                    last_operation_time = last_task.get_end() # start + duration

            # 2. check if the operation can fit in
            is_fit = False
            for (start, end) in idle_times:
                tb = np.max([start, last_operation_time])
                if tb + selected_machine.operation_time <= end:
                    # its fit :), lets put it in there
                    # print('insert', (start, end), tb, selected_machine.operation_time)
                    resource.add_task(operation, tb)
                    is_fit = True
                    break

            if not is_fit:
                # the operation is not fit in any idle time, so put it in the last operation
                
                last_resource_time = resource.get_last_operation_time()
                tb = np.max([last_resource_time, last_operation_time])
                # print('add_last', job_id, operation_id, '=>', last_resource_time, last_operation_time)
                resource.add_task(operation, tb)

            # increment the operation id for next operation
            current_job_operations[job_id - 1] += 1
        
        return resources

    def calculate_fitness(self, chromosome) -> int:
        resources = self.decode_chromosome(chromosome)
        makespan = 0
        for resource in resources:
            makespan = np.max([resource.get_last_operation_time(), makespan])
        return makespan

    def evaluate(self):
        for i in range(len(self.population)):
            fitness = self.calculate_fitness(self.population[i])
            self.population[i].set_fitness(fitness)
        
        # sort population based on fitness
        sorted(self.population, key=lambda c: c.fitness)

    def solve(self, problem: Problem, population_amount=100, gs=.6, ls=.3, rs=.1, parent_selector='tournament', pm=.1, iter=100, selected_offspring=.5) -> List[Resource]:
        self.problem = problem
        # print(self.global_selection().machine_selection)
        self.init_population(population_amount, gs, ls, rs)
        self.evaluate()

        selector = self.get_parent_selector(parent_selector)

        two_point_crossover = self.get_crossover("two_point")
        uniform_crossover = self.get_crossover("uniform")
        pox_crossover = self.get_crossover("pox")

        os_mutator = OSMutator(pm)
        ms_mutator = MSMutator(pm)

        print("========== Before =============")
        top_3 = self.population[:3]
        for i, c in enumerate(top_3):
            print(f"Top {i+1}")
            print("Machine Selection:", c.machine_selection)
            print("Operation Sequence:", c.operation_sequence)
            print("Fitness/Makespan:", c.fitness)
            print("==========================================")
        

        new_population: List[Chromosome] = []
        crossover_amount = 0
        mutation_amount = 0
        for i in range(iter):
            print("Generation", i+1)
            while (len(new_population) < population_amount):
                # select 2 parent
                p1_idx = selector.select_parent(self.population)
                p2_idx = selector.select_parent(self.population)
                p1 = self.population[p1_idx]
                p2 = self.population[p2_idx]

                if crossover_amount < 3:
                    print("Before Crossover")
                    print("Selected Parent 1:")
                    print("Machine Selection:", p1.machine_selection)
                    print("Operation Sequence:", p1.operation_sequence)
                    print("Selected Parent 2:")
                    print("Machine Selection:", p2.machine_selection)
                    print("Operation Sequence:", p2.operation_sequence)
                
                if len(new_population) < population_amount // 2:
                    ms1, ms2 = two_point_crossover.crossover(p1.machine_selection, p2.machine_selection)
                else:
                    ms1, ms2 = uniform_crossover.crossover(p1.machine_selection, p2.machine_selection)
                os1, os2 = pox_crossover.crossover(p1.operation_sequence, p2.operation_sequence)
                c1 = Chromosome(ms1, os1)
                c2 = Chromosome(ms2, os2)
                c1 = self.fix_chromosome(c1)
                c2 = self.fix_chromosome(c2)
                new_population.append(c1)
                new_population.append(c2)

                if crossover_amount < 3:
                    print("After Crossover")
                    print("Offspring 1:")
                    print("Machine Selection:", c1.machine_selection)
                    print("Operation Sequence:", c2.operation_sequence)
                    print("Offspring 2:")
                    print("Machine Selection:", c1.machine_selection)
                    print("Operation Sequence:", c2.operation_sequence)
                
                crossover_amount += 1
               
            
            for i, c in enumerate(new_population):
                # do mutation if p < pm
                p = np.random.rand()
                if p < pm:
                    if mutation_amount < 3:
                        print("Before Mutation")
                        print("Machine Selection:", c.machine_selection)
                        print("Operation Sequence:", c.operation_sequence)
                    ms = ms_mutator.mutate(c.machine_selection, self.problem)
                    os = os_mutator.mutate(c.operation_sequence)
                    new_population[i] = Chromosome(ms, os)
                    if mutation_amount < 3:
                        print("After Mutation")
                        print("Machine Selection:", ms)
                        print("Operation Sequence:", os)
                    mutation_amount += 1
            
            # self.population = new_population
            for i in range(len(new_population)):
                fitness = self.calculate_fitness(new_population[i])
                new_population[i].set_fitness(fitness)
            sorted(new_population, key=lambda c: c.fitness)

            # set top-t% from new population
            t = int(selected_offspring*population_amount)
            self.population[-t:] = new_population[:t]
            
            # re-evaluate the new population
            self.evaluate()
            best_chromosome = self.population[0]
            print("Best fitness:", best_chromosome.fitness)
        
        print("========== After ============")
        # get the best chromosome
        top_3 = self.population[:3]
        for i, c in enumerate(top_3):
            print(f"Top {i+1}")
            print("Machine Selection:", c.machine_selection)
            print("Operation Sequence:", c.operation_sequence)
            print("Fitness/Makespan:", c.fitness)
            print("==========================================")
        resources = self.decode_chromosome(best_chromosome)
        return resources
