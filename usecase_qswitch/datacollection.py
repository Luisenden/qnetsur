import sys

sys.path.append('../')
from src.datacollectors import ResultCollector, SurrogateCollector, AxCollector, RsCollector

class SaCollector(ResultCollector):
    def __init__(self, folder):
        super().__init__(folder, 'SA_')

    def get_total(self):
        print(self.data)
        return self.data

if __name__ == '__main__':

    folder = '../../surdata/qswitch/'
    coll = SurrogateCollector(folder)
    df = coll.get_final(coll.prefix+'qswitch6-30min')
    print(df)

    coll = AxCollector(folder)
    df = coll.get_final(coll.prefix+'qswitch6-30min')
    print(df)


    coll = SaCollector(folder)
    df = coll.get_final(coll.prefix+'qswitch6-30min')
    print(df)

    coll = RsCollector(folder)
    df = coll.get_final(coll.prefix+'qswitch6-30min')
    print(df)





