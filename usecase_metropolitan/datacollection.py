import sys
import numpy as np

sys.path.append('../')
from qnetsur.folderdatacollectors import ResultCollector, SurrogateCollector, AxCollector, SaCollector

class RsCollector(ResultCollector):
    def __init__(self, folder):
        super().__init__(folder, 'RS_')

    def get_total(self):
        self.data = self.data[0]
        self.data['objective'] = self.data['Utility'].apply(lambda x: sum(x))
        self.data['std'] = self.data['std'].apply(lambda x: np.sqrt(sum(x**2)))
        return self.data
    


if __name__ == '__main__':

    folder = '../../surdata/rb_budget_25h/'
    coll = SurrogateCollector(folder)
    df = coll.get_final(coll.prefix+'rb_starlight_budget_25h')

    coll = AxCollector(folder)
    df = coll.get_final(coll.prefix+'rb_starlight_budget_25h')
    print(df)


    coll = SaCollector(folder)
    df = coll.get_final(coll.prefix+'rb_starlight_budget_25h')
    print(df)

    coll = RsCollector(folder)
    df = coll.get_final(coll.prefix+'rb_starlight_budget_25h')
    print(df)





