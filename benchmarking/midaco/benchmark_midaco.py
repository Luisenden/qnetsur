
import midaco
from cocoex import Suite, Observer
import time



def problem_function(x):

    f = [0.0]*1 # Initialize array for objectives F(X)
    g = [0.0]*1 # Initialize array for constraints G(X)

    # Objective functions F(X)
    f[0] = fun(x)
    
    #  Equality constraints G(X) = 0 MUST COME FIRST in g[0:me-1]
    # Inequality constraints G(X) >= 0 MUST COME SECOND in g[me:m-1] 
    g[0] = 0
    
    return f, g



def midaco_optimize(problem, lb, ub, D):
    
  key = b'Luise_Prielinger___[TRIAL-LICENSE-valid-until-1-August-2025]'
  problem = {} # Initialize dictionary containing problem specifications
  option  = {} # Initialize dictionary containing MIDACO options


  problem['@'] = problem_function
  problem['o'] = 1
  problem['m'] = 1
  problem['n'] = D
  problem['ni'] = 0
  problem['xl'] = lb
  problem['xu'] = ub
  problem['x'] = lb
  problem['me'] = 0


  option['maxeval'] = 10000000     # Maximum number of function evaluation (e.g. 1000000) 
  option['maxtime'] = 60*60*24  # Maximum time limit in Seconds (e.g. 1 Day = 60*60*24) 
  option['printeval'] = 100000  # Print-Frequency for current best solution (e.g. 1000) 
  option['save2file'] = 0     # Save SCREEN and SOLUTION to TXT-files [0=NO/1=YES]
  option['parallel'] = 0 # Serial: 0 or 1, Parallel: 2,3,4,5,6,7,8... 

  option['param1']  = 0.0  # ACCURACY  
  option['param2']  = 0.0  # SEED  
  option['param3']  = 0.0  # FSTOP  
  option['param4']  = 0.0  # ALGOSTOP  
  option['param5']  = 0.0  # EVALSTOP  
  option['param6']  = 0.0  # FOCUS  
  option['param7']  = 0.0  # ANTS  
  option['param8']  = 0.0  # KERNEL  
  option['param9']  = 0.0  # ORACLE  
  option['param10'] = 0.0  # PARETOMAX
  option['param11'] = 0.0  # EPSILON  
  option['param12'] = 0.0  # BALANCE
  option['param13'] = 0.0  # CHARACTER

  solution = midaco.run( problem, option, key )

  return solution['f']



if __name__ == '__main__': 


  suite = Suite("bbob", "year: 2009", f"dimensions: 2,3,5,10,20,40")

  observer = Observer("bbob", "result_folder: %s_on_%s" % (midaco_optimize.__name__, "bbob2009"))
  solver = midaco_optimize

  count = 0
  for fun in suite:
    
    fun.observe_with(observer)
    start= time.time()
    solver(fun, fun.lower_bounds, fun.upper_bounds, fun.dimension)
    end = time.time()
    print(f"Function {count} of {len(suite)} done in {end-start} seconds.")

    count += 1
