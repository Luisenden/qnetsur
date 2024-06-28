What is qnetsur?
================
 `qnetsur` presents a tool to optimize parameters of expensive quantum-network simulations. It implements a surrogate-optimization technique based on simple regression models in `sklearn` - [`DecisionTreeRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#decisiontreeregressor) and [`SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#svr).

The Context
-----------
To bring quantum networks closer to real-world use, it's important to deeply understand both the hardware and the protocols involved. As these physical systems grow more complex, traditional analytical methods often aren't enough. Researchers rely on detailed numerical simulations to study quantum-network behavior e.g., using [NetSquid](https://netsquid.org). However, these simulations can be very complex and demanding, and their exact workings are often unknown. This makes it hard to use traditional optimization methods that assume the function is smooth or predictable.

The Challenge
-------------
Quantum network simulations are computationally intense. Global optimization methods like simulated annealing or genetic algorithms require many function evaluations, which is impractical for such heavy computations. `qnetsur` addresses this by using smart machine learning models to make the optimization process more efficient and feasible.

