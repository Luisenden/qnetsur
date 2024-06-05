import sys
sys.path.append('../')
from src.folderdatacollectors import ResultCollector, SurrogateCollector, AxCollector, RsCollector

class SaCollector(ResultCollector):
    def __init__(self, folder):
        super().__init__(folder, 'SA_')

    def get_total(self):
        print(self.data)
        return self.data

if __name__ == '__main__':

    folder = '../../surdata/cd_tree-2-1/'
    folder = '../../surdata/cd_randtree-20/'
    folder = '../../surdata/cd_randtree-100-10h/'
    sur = SurrogateCollector(folder)
    df = sur.get_final(sur.prefix+'randtree-100_10h')
    print(df)

    sa = SaCollector(folder)
    df = sa.get_final(sa.prefix+'randtree-100_10h')
    print(df)

    ax = AxCollector(folder)
    df = ax.get_final(ax.prefix+'randtree-100_10h')
    print(df)

    rs = RsCollector(folder)
    df = rs.get_final(rs.prefix+'randtree-100_10h')
    print(df)





