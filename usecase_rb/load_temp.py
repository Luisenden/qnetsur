from plotting_tools import *

folder = '../../surdata/rb_budget'

# #plot results of optimization (Utility)
# plot_optimization_results(folder)

# # plot from exhaustive run
# plot_from_exhaustive(folder)

# # plot time profiling
# time_profile, rel_time_profile = read_pkl_surrogate_timeprofiling(folder)
# print(rel_time_profile.mean(axis=0))

# df = get_performance_distribution_per_method(folder)
# print(df)

error, acquisition = read_pkl_surrogate_benchmarking(folder)
print(error, acquisition)

