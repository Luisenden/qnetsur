from config import *
from src.utils import *

from simulation import *


if __name__ == '__main__':

        start = time.time()

        s = Surrogate(simulation_rb, vals=vals, vars=vars, initial_model_size=5)
        s.optimize(MAXITER=1, verbose=True)
        print(s.X_df)
        print(s.y)

        print('time:', time.time()-start)