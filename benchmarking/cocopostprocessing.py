import cocopp  

cocopp.genericsettings.xlimit_pprldmany = 1e3 # set to 1e3 for quick tests
result_folder = "/Users/localadmin/Library/CloudStorage/OneDrive-DelftUniversityofTechnology/1_backup_project_surrogate/benchmark_output/exdata_02-03-25/sur_optimize_on_bbob2009-0004"
cocopp.main(result_folder + ' 13/cga-grid100!' + ' 23/m-sdwoa!' + ' 12/cmaes!' + ' 19/nelder-mead-scipy!');  # re-run folders look like "...-001" etc