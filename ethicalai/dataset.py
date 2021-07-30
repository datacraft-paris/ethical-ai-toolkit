
import pandas as pd
import plotly.express as px
from ipywidgets import interact
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

class ClassificationDataset:
    def __init__(self,data,target):

        self._data = data
        self._target = target


    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target


    def plot(self,col,barmode = "group",nbins = 5,norm = False,**kwargs):
        assert col in self.data.columns

        if not norm:
            return px.histogram(self.data,y = col,color = self.target,barmode = barmode,nbins = nbins,**kwargs)
        else:

            if is_numeric_dtype(self.data[col]):

                data_col = self.data.copy()
                data_col[col] = pd.qcut(data_col[col],q = nbins).astype(str)
            
            else:
                data_col = self.data.copy()

            data_col = (data_col
                .assign(count = lambda x : 1)
                .groupby([col,self.target],as_index = False)
                ["count"]
                .sum()
            )

            data_col["percent"] = data_col.groupby(col)["count"].transform(lambda x : x / x.sum())

            return px.bar(data_col,y = col,x = "percent",color = self.target,barmode = "stack",**kwargs)


    def notebook_plot(self,**kwargs):

        @interact(col = self.data.columns)
        def show(col):
            return self.plot(col,**kwargs)
    
