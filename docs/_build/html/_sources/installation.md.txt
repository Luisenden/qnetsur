Installation
============

Before installation, make sure your system satisfies the following prerequisites
* Python version 3.7 or higher.
* Linux or MacOS operating system.
* We recommend using virtual environemnts e.g., [miniconda](https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux>).
* The python package installer [pip](https://pip.pypa.io/en/stable/). To upgrade your current pip version run `pip install --upgrade pip`.

Open a terminal, activate your virtual environment and execute
```python
pip install git+https://github.com/Luisenden/qnetsur
pip install -r requirements.txt
```

Tutorial
========

This [notebook](https://drive.google.com/file/d/1L5JqMcPL0rKUI3ROri21oK287ozlNUQA/view?usp=sharing) (google colab account required) will help you get up to speed with `qnetsur`. 
It features the optimization of parameters of a quantum network protocol that is based on continuous-entanglement distribution. Check it out!

About the performance of a quantum network
------------------------------------------
Consider a vector function $f(\mathbf{x}): X \rightarrow \mathbb{R}^{m_f}$ that describes a performance metric of a quantum network, like the average fidelity of quantum states between node pairs. This function takes parameter values $\mathbf{x} = \{x_1, x_2, \ldots, x_N\}$ as input. Based on $f$, we can define an objective function $U(f)$ which reflects the network's perceived utility. Essentially, while $f(\mathbf{x})$ describes the network's output, the function $U(f, \mathbf{x}): \mathbb{R}^{m_f} \times X \rightarrow \mathbb{R}^m$ measures the utility derived from this output. Though $U$ can represent any objective, in our use cases, an element $U^{(i)}$ represents the utility perceived by a user $i$. 

**Utility can be defined based on different things, typical examples are**
* Distillable entanglement
* Secret key rate
* Number of completed user requests
* etc.

Network parameters
------------------
A network parameter $x_p$ is defined on a domain $X_p$. Together, these domains form the input space: $X = X_1 \times X_2 \times \cdots \times X_N$. The parameter $x_p$ is not restricted to a specific datatype. For continuous and discrete values, $x_p$ is within minimum-maximum bounds $X_p = [x_p^{\mathrm{min}}, x_p^{\mathrm{max}}]$, while ordinal and categorical parameters are vectors.

Probabilistic processes
-----------------------
Many processes in a quantum network are probabilistic. For example, creating entanglement between nodes often requires multiple independent attempts, following a geometric distribution. Hence, we assume $f$ and the utility $U(f, \mathbf{x})$ are stochastic functions. The utility perceived by a user $i$ is a random variable $Y^{(i)}$, following the distribution $Y^{(i)} \sim U^{(i)}(f, \mathbf{x})$ with expectation $E[Y^{(i)}]$. Evaluating $f$ via numerical simulation provides a finite empirical sample $\{ U^{(i)}_1, U^{(i)}_2, \dots, U^{(i)}_n\}$, used to estimate $E[Y^{(i)}]$ for each user $i$ with the sample mean $\overline{U}^{(i)}(\mathbf{x}) = \frac{1}{n} \sum_{j=1}^n U^{(i)}(f, \mathbf{x})_j$.

Workflow
========
Simplified surrogate-assisted workflow: 
1. Importing a quantum-network simulation
2. Generate $k_0$ different network configurations $\{\textbf{x}_1, \ldots, \textbf{x}_{k_0}\}$ from the search domain $X$. 
3. Execute quantum network simulation using the latter assembles the initial training data of parameter sets along with their means $[\overline{U}^{(1)}(\mathbf{x}_i), \ldots, \overline{U}^{(m)}(\mathbf{x}_i)]$. 
4. Machine learning models train on the dataset and their performance is evaluated (using five-fold cross validation on current data set). 
5. In the acquisition process, the better performing model predicts utility values for a large number of points close to the currently best performing configurations.
6. The parameter sets  associated with the highest predicted utility get passed to the simulation.
7. Execute simulations and appended configurations and outcomes to training set
8. Repeat from 4. until the optimization concludes upon reaching a maximum time or number of cycles $T$.

![workflow](Figures/workflow.pdf)