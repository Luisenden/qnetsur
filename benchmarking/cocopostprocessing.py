import cocopp  

cocopp.compall.pprldmany.divide_by_dimension = False
cocopp.compall.pprldmany.display_best = False
cocopp.genericsettings.xlimit_pprldmany = 1e4

# clean
result_folder = "exdata/sur_optimize_on_bbob2009-d10 midaco/exdata/midaco_optimize_on_bbob2009"
cocopp.main(result_folder + ' bbob/2009/ga!' + ' bbob/2009/eda!' + ' bbob/2009/poems!');  # re-run folders look like "...-001" etc

# noisy
result_folder = "exdata/sur_optimize_on_bbob2009-d10-noisy midaco/exdata/midaco_optimize_on_bbob2009-noisy"
cocopp.main(result_folder + ' bbob-noisy/2009/dasa!' + ' bbob-noisy/2009/bfgs!' + ' bbob-noisy/2009/eda!');  # re-run folders look like "...-001" etc