# Surrogate optimization for quantum networks
 `qnetsur` presents a tool to optimize parameters of expensive quantum-network simulations. It is based on simple regression models in `sklearn` ([DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#decisiontreeregressor) and [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#svr)).

Context: In order to bring quantum networks closer to reality, we need to thoroughly understand quantum network hardware and
associated protocols. As physical architectures quickly become too complex for analytical study, the research field widely
relies on comprehensive numerical simulations to investigate quantum-network behavior. These simulations can be highly
informative, but their functional form is typically unknown. When it comes to optimization, techniques relying on assumptions
about the function’s continuity, differentiability, or convexity are thus inapplicable. Additionally, quantum network simulations
are computationally demanding, rendering global approaches like simulated annealing or genetic algorithms – which require
extensive function evaluations – impractical.


## Requirements 
* Python 3.7+
* scipy
* scikit-learn==1.3.1
* numpy
* pandas

## Getting started
```python
pip install git+https://github.com/Luisenden/qnetsur
```
