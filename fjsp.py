from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Machine:
    def __init__(self, id: int, operation_time: int=-1):
        self.id = id
        self.index = id - 1
        self.operation_time = operation_time

class Operation:
    def __init__(self, id: int, job_id: int):
        self.job_id = job_id
        self.id = id
        self.index = id - 1
        self.machines: List[Machine] = []

    def add_machine(self, machine: Machine):
        self.machines.append(machine)

    def get_machine_by_id(self, id: int) -> Machine:
        for m in self.machines:
            if m.id == id:
                return m
        return None

    def get_random_machine(self, return_index=False):
        idx = np.random.randint(0, len(self.machines))
        if return_index:
            return idx
        else:
            return self.machines[idx]
    
    def get_lowest_machine(self) -> int:
        lowest_idx = 0
        for i, m in enumerate(self.machines):
            if m.operation_time < self.machines[lowest_idx].operation_time:
                lowest_idx = i
        return lowest_idx

class Task:
    def __init__(self, operation: Operation, relative_start: int, duration: int):
        self.operation = operation
        self.relative_start = relative_start
        self.duration = duration
    
    def get_end(self) -> int:
        return self.relative_start + self.duration

class Resource:
    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.tasks: List[Task] = []
    
    def add_task(self, operation: Operation, relative_start: int):
        used_machine = operation.get_machine_by_id(self.machine_id)
        if used_machine == None:
            raise ValueError("Machine on operation and task is not match")
        # insert in sorted manner for easier to calculate
        insert_pos = 0
        for i, task in enumerate(self.tasks):
            if task.relative_start < relative_start:
                insert_pos = i + 1
        self.tasks.insert(insert_pos, Task(operation, relative_start, used_machine.operation_time))
        if self.is_conflict():
            # for t in self.tasks:
            #     print(t.relative_start, t.get_end(), t.duration)
            raise ValueError(f"The operation {operation.job_id}-{operation.id} is conflict. relative start: {relative_start}. duration: {used_machine.operation_time}")

    def add_last(self, operation: Operation):
        last_operation_time = self.get_last_operation_time()
        self.add_task(operation, last_operation_time)

    def is_conflict(self) -> bool:
        current_timestamp = 0
        for task in self.tasks:
            if task.relative_start < current_timestamp: # overlapping
                # print('Conflict detail:', task.relative_start, '<=', current_timestamp, 'job id:', task.operation.job_id, 'opr id:', task.operation.id)
                return True
            current_timestamp = task.get_end()
        return False

    def find_idle_time(self) -> List[Tuple[int, int]]:
        idle_time: List[Tuple(int, int)] = [] # tuple(start, end) of idle time
        current_timestamp = 0
        for task in self.tasks:
            if np.abs(task.relative_start - current_timestamp) > 0: # there's an idle time
                idle_time.append((current_timestamp, task.relative_start))
            current_timestamp = task.get_end()
        return idle_time

    def get_last_operation_time(self):
        if len(self.tasks) == 0:
            return 0
        task = self.tasks[-1]
        return task.get_end()

    def find_operation(self, job_id: int, operation_id: int) -> Task:
        for task in self.tasks:
            if task.operation.job_id == job_id and task.operation.id == operation_id:
                return task
        return None

class Job:
    def __init__(self, id: int):
        self.id = id
        self.index = id - 1
        self.operations: List[Operation] = []

    def add_operation(self, operation: Operation):
        self.operations.append(operation)

    def get_operation_by_id(self, id: int) -> Operation:
        for opr in self.operations:
            if opr.id == id:
                return opr
        return None


class Problem:
    def __init__(self, jobs: List[Job], n_machine: int):
        self.jobs = jobs
        self.n_machine = n_machine
        self.L = self.get_all_operation_count()

    def get_all_operation_count(self):
        L = 0
        for job in self.jobs:
            L += len(job.operations)
        return L

    def get_operation_by_index(self, idx) -> Operation:
        i = 0
        for job in self.jobs:
            for opr in job.operations:
                if i == idx:
                    return opr
                i += 1
        return None

    def select_random_job(self, return_index=False):
        idx = np.random.randint(0, len(self.jobs))
        if return_index:
            return idx
        else:
            return self.jobs[idx]
    
    def get_job(self, index) -> Job:
        assert index < 0 or index >= len(self.jobs), 'Index is out of range'
        return self.jobs[index]

    def get_job_by_id(self, id: int) -> Job:
        for job in self.jobs:
            if job.id == id:
                return job
        return None

    def get_shuffled_job(self) -> List[Job]:
        job_copy = np.copy(self.jobs)
        np.random.shuffle(job_copy)
        return job_copy

class Solver:
    def __init__(self, name: str):
        self.name = name

    def solve(self, problem: Problem, **kwargs):
        self.problem = problem
                

class FJSP:
    def __init__(self, dataset: str, solver: Solver):
        self.solver = solver
        self.problem: Problem = self.parse_dataset(dataset)
    
    def solve(self, **kwargs) -> List[Resource]:
        return self.solver.solve(self.problem, **kwargs)

    def parse_dataset(self, dataset) -> Problem:
        with open(dataset, "r") as file:
            number_job, number_machine = file.readline()[:-1].split(" ")[:2]
            number_job = int(number_job)
            number_machine = int(number_machine)

            jobs: list[Job] = []
            for i, row in enumerate(file.readlines()):
                if i >= number_job:
                    break
                data = row[:-1].split(" ")
                n_operation = int(data[0])
                job = Job(i+1)
                pointer = 1
                while pointer < len(data):
                    n_machine = int(data[pointer])
                    pointer += 1
                    operation = Operation(len(job.operations) + 1, job.id)
                    for m in range(n_machine):
                        machine_id = int(data[pointer])
                        if machine_id > number_machine:
                            raise ValueError(f"Machine id: {machine_id} is not available. max machine: {number_machine}")
                        pointer += 1
                        operation_time = int(data[pointer])
                        pointer += 1
                        operation.add_machine(Machine(machine_id, operation_time))
                    job.add_operation(operation)
                if len(job.operations) != n_operation:
                    raise ValueError(f"Job_id: {job.id}. n_operation should be {n_operation}. {len(job.operations)} found!")
                jobs.append(job)
            
        return Problem(jobs, number_machine)

def save_as_fig(filename: str, resources: List[Resource], width=100, height=9):
    data = {
        'resource': [],
        'start': [],
        'duration': [],
        'color': [],
        'label': []
    }
    colors = [mcolors.CSS4_COLORS[k] for k in mcolors.CSS4_COLORS if k.lower() != 'white']
    for resource in resources:
        for task in resource.tasks:
            data['resource'].append(f"Machine {resource.machine_id}")
            data['start'].append(task.relative_start)
            data['duration'].append(task.duration)
            data['color'].append(colors[task.operation.job_id % len(colors)].lower())
            data['label'].append(f"J{task.operation.job_id}.{task.operation.id}")
    
    plt.figure(figsize=(width, height))
    plt.barh(y=data['resource'], left=data['start'], width=data['duration'], color=data['color'])

    # Invert y axis
    plt.gca().invert_yaxis()

    for i in range(len(data['label'])):
        plt.text(data['start'][i], data['resource'][i], data['label'][i])

    # add grid lines
    plt.grid(axis='x', alpha=0.5)

    #save fig
    plt.savefig(filename)