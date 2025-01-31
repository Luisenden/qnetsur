# import cocoex  # experimentation module
import cocopp  # post-processing module (not strictly necessary)
# import scipy  # to define the solver to be benchmarked



# ### input
# suite_name = "bbob"
# fmin = scipy.optimize.fmin  # optimizer to be benchmarked
# budget_multiplier = 1  # x dimension, increase to 3, 10, 30,...

# ### prepare
# suite = cocoex.Suite(suite_name, "", "")  # see https://numbbo.github.io/coco-doc/C/#suite-parameters
# output_folder = '{}_of_{}_{}D_on_{}'.format(
#         fmin.__name__, fmin.__module__ or '', int(budget_multiplier), suite_name)
# observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
# repeater = cocoex.ExperimentRepeater(budget_multiplier)  # 0 == no repetitions
# minimal_print = cocoex.utilities.MiniPrint()

# print(help(suite))

### go
# while not repeater.done():  # while budget is left and successes are few
#     for problem in suite:  # loop takes 2-3 minutes x budget_multiplier
#         print(help(problem))
#         if repeater.done(problem):
#             continue  # skip this problem
#         problem.observe_with(observer)  # generate data for cocopp
#         problem(problem.dimension * [0])  # for better comparability
#         xopt = fmin(problem, repeater.initial_solution_proposal(problem),
#                     disp=False)
#         problem(xopt)  # make sure the returned solution is evaluated
#         repeater.track(problem)  # track evaluations and final_target_hit
#         minimal_print(problem)  # show progress

# ### post-process data
# cocopp.main(observer.result_folder + ' bfgs!');  # re-run folders look like "...-001" etc

import numpy as np
from cocoex import Suite, Observer


MAX_FE = 10000  # max f-evaluations
def random_search1(f, lb, ub, m):  # don't use m >> 1e5 with this implementation
    candidates = lb + (ub - lb) * np.random.rand(m, len(lb))
    return candidates[np.argmin([f(x) for x in candidates])]


solver1 = random_search1
count = 0
suite = Suite("bbob", "year: 2009", "dimensions: 10")  # https://numbbo.github.io/coco-doc/C/#suite-parameters
observer = Observer("bbob", "result_folder: %s_on_%s" % (solver1.__name__, "bbob2009"))
for fun in suite:
    assert fun.evaluations == 0
    # if fun.dimension > 10:
    #     break
    print('Current problem index = %d' % fun.index)
    fun.observe_with(observer)
    assert fun.evaluations == 0
    solver1(fun, fun.lower_bounds, fun.upper_bounds, MAX_FE)
    count += 1
    # data should be now in the "exdata/random_search_on_bbob2009" folder
    # assert fun.evaluations == MAX_FE  # depends on the solver
    # doctest: +ELLIPSIS

# ### post-process data
# cocopp.main(observer.result_folder + ' bfgs!');  # re-run folders look like "...-001" etc
print(count)