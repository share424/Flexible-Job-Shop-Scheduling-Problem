# Flexible Job-Shop Scheduling Problem
An implementation of [An effective genetic algorithm for the flexible job-shop scheduling problem](https://www.sciencedirect.com/science/article/abs/pii/S095741741000953X) paper

## Requirements
install the dependencies by
```
$ pip install -r requirements.txt
```

## How to use?
import the `FJSP` module and the solver module. Currently only available GA for the solver

Example
```python
from fjsp import FJSP, save_as_fig
from ga import GeneticAlgoritm

solver = GeneticAlgoritm()
fjsp = FJSP("dataset.fjs", solver)
resources = fjsp.solve(iter=10, selected_offspring=.7)
save_as_fig('output.png', resources)
```

## Example Output
![](https://github.com/share424/Flexible-Job-Shop-Scheduling-Problem/blob/master/test.png?raw=true)

## Documentations
### GeneticAlgorithm.solve(problem: Problem, population_amount=100, gs=.6, ls=.3, rs=.1, parent_selector='tournament', pm=.1, iter=100, selected_offspring=.5) -> List[Resource]
solve the genetic algoritm based on the given problem. Return the decoded best chromosome
| Properties       | Description     | Default     |
| :------------- | :----------: | -----------: |
|  `problem` | The problem that need to be solved   |     |
|  `population_amount` | the initial population amount   | `100`    |
| `gs`   | The fragment amount of global selection for initialitation population | `0.6` |
| `ls`   | The fragment amount of local selection for initialitation population | `0.3` |
| `rs`   | The fragment amount of random selection for initialitation population | `0.1` |
| `parent_selector`   | Parent selection strategy. available value `tournament` and `roullete_wheel` | `tournament` |
| `pm`   | Mutation probability | `0.1` |
| `iter`   | Amount of generation | `100` |
| `selected_offspring`   | Amount of selected offspring to replace current generation | `0.5` |

**Note: make sure `gs + ls + rs == 1`, otherwise an error will thrown**

### GeneticAlgorithm.decode_chromosome(chromosome: Chromosome) -> List[Resource]
Decode the given chromosome into list of resources
| Properties       | Description     | Default     |
| :------------- | :----------: | -----------: |
|  `chromosome` | the given chromosome   |     |

### GeneticAlgorithm.calculate_fitness(chromosome: Chromosome) -> int
Calculate the fitness of the chromosome. In this problem, this function will calculate the makespan of the problem
| Properties       | Description     | Default     |
| :------------- | :----------: | -----------: |
|  `chromosome` | the given chromosome   |     |

### GeneticAlgorithm.evaluate() -> int
Calculate all fitness of the current population and sort them based on the best fitness (ascending)

### GeneticAlgorithm.global_selection() -> int
Do global selection to get 1 chromosome

### GeneticAlgorithm.local_selection() -> int
Do local selection to get 1 chromosome

### GeneticAlgorithm.random_selection() -> int
Do random selection to get 1 chromosome

### GeneticAlgorithm.is_valid_chromosome(chromosome: Chromosome) -> bool
Check if the given chromosome is valid or not. This function will check the machine selection part to check if the selected machine is available in the operation or not
| Properties       | Description     | Default     |
| :------------- | :----------: | -----------: |
|  `chromosome` | the given chromosome   |     |

### GeneticAlgorithm.fix_chromosome(chromosome: Chromosome) -> Chromosome
Fix the invalid chromosome from `is_valid_chromosome`. This function will set the invalid machine with the last available machine in that operation
| Properties       | Description     | Default     |
| :------------- | :----------: | -----------: |
|  `chromosome` | the given chromosome   |     |

### FJSP.__init__(dataset: str, solver: Solver)
Read the dataset and set the solver
| Properties       | Description     | Default     |
| :------------- | :----------: | -----------: |
|  `dataset` | the dataset file in `.fjs` format   |     |
|  `solver` | the solver   |     |

### FJSP.solve(**kwargs)
solve the given problem. This function will call the solver `solve` function

### fjsp.save_as_fig(filename: str, resources: List[Resource], width=100, height=9)
save the result from `solve` function to `gantt` chart
| Properties       | Description     | Default     |
| :------------- | :----------: | -----------: |
|  `filename` | the output file   |     |
|  `resources` | the resources result from the solver   |     |
|  `width` | the figure height   |  `100`   |
|  `height` | the figure width   |  `9`   |

## Dataset
* in the first line there are (at least) 2 numbers: the first is the number of jobs and the second the number of machines (the 3rd is not necessary, it is the average number of machines per operation)

* Every row represents one job: the first number is the number of operations of that job, the second number (let's say k>=1) is the number of machines that can process the first operation; then according to k, there are k pairs of numbers (machine,processing time) that specify which are the machines and the processing times; then the data for the second operation and so on...


Example: Fisher and Thompson 6x6 instance, alternate name (mt06)
```
6   6   1   
6   1   3   1   1   1   3   1   2   6   1   4   7   1   6   3   1   5   6   
6   1   2   8   1   3   5   1   5   10  1   6   10  1   1   10  1   4   4   
6   1   3   5   1   4   4   1   6   8   1   1   9   1   2   1   1   5   7   
6   1   2   5   1   1   5   1   3   5   1   4   3   1   5   8   1   6   9   
6   1   3   9   1   2   3   1   5   5   1   6   4   1   1   3   1   4   1   
6   1   2   3   1   4   3   1   6   9   1   1   10  1   5   4   1   3   1   
```
first row = 6 jobs and 6 machines 1 machine per operation
second row: job 1 has 6 operations, the first operation can be processed by 1 machine that is machine 3 with processing time 1.

## License
MIT