import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import panel as pn
from pyforbes import ForbesList
import hvplot.pandas
pn.extension('tabulator')

flist = ForbesList()
years = []
df = pd.DataFrame()
for i in range(1997, 2023):
    years.append(i)
df["year"] = years

#Create an array to keep track of the total number of billionaires each year
number_billionaires = []
for i in range(1997,2023):
    number_df = flist.get_df("billionaires", year =i)
    number_billionaires.append(number_df.shape[0])
df["Total Billionaires"] = number_billionaires

#Enable the DataFrame to be interactive
idf = df.interactive()

        ### World map visualisation
import folium as fm
import pandas as pd
import param
import panel as pn
import random
pn.extension(sizing_mode="stretch_width")

def get_map():
    return fm.Map()

mmap = get_map()

pn.panel(mmap, height=400)

url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
country_shapes = f'{url}/world-countries.json'


#building the data

df_years = pd.read_csv("data/number_billionaires_by_year.csv",sep=";")
def get_df(year = 2000):
    
    return df_years[df_years.year == year]

def update_map(mmap,df):
    fm.Choropleth(
    #The GeoJSON data to represent the world country
    geo_data=country_shapes,
    name='Number of Billionaires per country',
    data=df.drop("year",axis=1),#year axis is unnessesary right now since it was already filtered out
    #The column aceppting list with 2 value; The country name and  the numerical value
    columns=['country', 'number'],
    key_on='feature.properties.name',
    fill_color='PuRd',
    nan_fill_color='white').add_to(mmap)
    

pn.panel(mmap, height=5000)

class PanelFoliumMap(param.Parameterized):
    year_count = param.Integer(2000, bounds=(1997,2022))
        
    def __init__(self, **params):
        super().__init__(**params)
        self.mmap = get_map()
        self.folium_pane = pn.pane.plot.Folium(sizing_mode="stretch_both", min_height=500, margin=0)    
        self.view = pn.Column(
            #self.param.year_count,
            self.folium_pane,
            sizing_mode="stretch_both", height=500
        )
        self._update_map()

    @param.depends("year_count", watch=True)
    def _update_map(self):
        self.mmap = get_map()
        df = get_df(year=self.year_count)
        update_map(self.mmap,df)
        self.folium_pane.object = self.mmap
        

        
myapp = PanelFoliumMap()

#Create the year slider widget
year_slider = myapp.param.year_count


#Setting up pipeline for total billionaires
total_pipeline = (
    idf[
        idf.year <= year_slider
    ]
    .reset_index()
    .sort_values(by='year')  
    .reset_index(drop=True)
)

#Creating plot for total billionaires using hvplot
total_billionaire_plot = total_pipeline.hvplot(x = 'year', y = 'Total Billionaires', title="Number of billionaires per year")


###Bar Plot section
bar_df = pd.DataFrame()
year_data = []
worth_data = []
name_data = []
for i in range(1997,2023):
    finalworth_df = flist.get_df("billionaires", year =i)
    finalworth_df = finalworth_df[['year', 'finalWorth', 'person', 'name']]
    finalworth_df = finalworth_df.sort_values(by='finalWorth',ascending=False)
    counter = 0
    for row in finalworth_df.iterrows():
        year_data.append(row[1].year)
        worth_data.append(row[1].finalWorth)
        name_data.append(row[1].person["name"])
        counter = counter + 1
        if (counter == 10):
            break


bar_df["year"] = year_data
bar_df["finalWorth"] = worth_data
bar_df["name"] = name_data

#making it interactive
bar_idf = bar_df.interactive()
#sorting it for display
bar_pipeline = (
    bar_idf[
        bar_idf.year == year_slider
    ]
    .reset_index()
    .sort_values(by='finalWorth')  
    .reset_index(drop=True)
)
worth_bar_plot = bar_pipeline.hvplot.barh(x = 'name', y = 'finalWorth', 
                                    title = 'Top 10 Billionaires Based on Final Worth')

        ###Plot of the Wealth of Billionaires every year coloured by ranking
from pyforbes import ForbesList
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
flist = ForbesList()
richest400_df = flist.get_df("billionaires", year =2022)

#for the billionaire wealth/rank plot
import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.colors as co
import matplotlib.cm as cm
#we will use this function on every loaded dataset to have consistent names in each dataFrame 

def uri_to_name(data):
    lst = []
    for index, row in data.iterrows():
        lst.append(row['person']["name"])
    data.name = pd.Series(lst)
uri_to_name(richest400_df)
richest400_df.head()
n = list(richest400_df.name)

def get_rank_wealth(name,year): #gives the rank and the wealth of the billionaire a certain year
    df = flist.get_df("billionaires", year =year)
    uri_to_name(df)
    
    if name in df.name.values:
        num_billionaires = df.shape[0]
                #this is the min max scaling of the rank of each billionaire by year
                #max+1               -  value                                         - min / max - min
        rank = ((num_billionaires+1) - int(df[df.name == name].finalWorth.index[0]+1) - 1)/(num_billionaires - 1)
        
        wealth = int(df[df.name == name].finalWorth.values[0])
        return rank,wealth
    else:#if the name isnt in this year returns rank 0 and wealth 0
        return 0,0

#Called rank_ column isntead of rank, it caused a weird bug
def list_of_wealth(name,start=1997,stop=2022):
    rank_wealth = pd.DataFrame(columns=["rank_","wealth"])

    for year in range(start,stop+1):
        
        rank,wealth = get_rank_wealth(name,year)
        #print(year,rank,wealth)
        #rank_wealth = rank_wealth.append({"rank_":rank,"wealth":wealth,"year":year},ignore_index=True)
        rank_wealth = rank_wealth.append({"rank_":rank,"wealth":wealth},ignore_index=True)

    return rank_wealth

def color_map_color(value, cmap_name='inferno', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    color = matplotlib.colors.rgb2hex(rgb)
    return color

def make_plot(Billionaire):
    bil=list_of_wealth(Billionaire)
    bil

    #return bil.hvplot(x = 'year', y = 'wealth', value_label='rank', title="Number of billionaires per year")

    print(Billionaire)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.subplots()
    
    y,r = bil.wealth.values,bil.rank_.values
    x = np.arange(1997,2023)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors

    #lc = LineCollection(segments, cmap='prism',colors = cm.hot(list(r)))
    c = list(map(color_map_color,r))
    lc = LineCollection(segments, cmap='inferno', color = c)
    # Set the values used for colormapping
    lc.set_array([0,1])
    lc.set_linewidth(2)
    lc.set_label('Line Number')
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    

    ax.set_title(f"{Billionaire}'s Wealth colored by comparative rank")

    ax.set_facecolor((0.0, 0.0, 0.0, 0.2))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(1, 200000)

    return fig


interact = pn.interact(make_plot, Billionaire=n)


    ### Dashboard
    
template = pn.template.FastListTemplate(
    title='Forbes Billionaires Data Visualizations', 
    sidebar=[pn.pane.Markdown("## Settings"),   
            myapp.param.year_count,
            interact[0]],
    main=[pn.Row(pn.Column(total_billionaire_plot.panel(width=700), margin=(0,25))),
                pn.Row(pn.Column(worth_bar_plot.panel(width=700), margin=(0,25))),
         pn.Row(myapp.view),
         pn.Row(interact[1])],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)
# template.show()
template.servable()























