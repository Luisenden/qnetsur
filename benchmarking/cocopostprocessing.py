import cocopp
import cocopp.compall
import cocopp.compall.pprldmany  

cocopp.compall.pprldmany.divide_by_dimension = False
cocopp.compall.pprldmany.displaybest = False
cocopp.genericsettings.xlimit_pprldmany = 1e4
cocopp.genericsettings.outputdir = "..."

print(cocopp.compall.pprldmany.displaybest)

folder = "path/to/data/benchmarking/exdata_all/"
midaco_folder = "path/to/data/benchmarking/exdata_all/midaco_optimize_on_bbob2009"

# clean
for i in [2, 3, 5, 10, 20]:
    sur_folder = folder+f"sur_optimize_on_bbob2009-d{i}"
    meta_folder = ' '+folder+f"meta_optimize_on_bbob2009-d{i}"
    simanneal_folder = ' '+folder+f"simanneal_optimize_on_bbob2009-d{i}"

    cocopp.main(sur_folder + meta_folder + simanneal_folder + midaco_folder + ' bbob/2009/ga!' + ' bbob/2009/eda!' + ' bbob/2009/poems!');  # re-run folders look like "...-001" etc

    # noisy
    sur_folder_noisy = folder+f"sur_optimize_on_bbob2009-d{i}-noisy"
    meta_folder_noisy = ' '+folder+f"meta_optimize_on_bbob2009-d{i}-noisy"
    simanneal_folder_noisy = ' '+folder+f"simanneal_optimize_on_bbob2009-d{i}-noisy"
    midaco_folder_noisy = ' '+folder+f"midaco_optimize_on_bbob2009-d{i}-noisy"
    cocopp.main(sur_folder_noisy +  meta_folder_noisy + simanneal_folder_noisy + midaco_folder_noisy + ' bbob-noisy/2009/dasa!' + ' bbob-noisy/2009/bfgs!' + ' bbob-noisy/2009/eda!');  # re-run folders look like "...-001" etc
