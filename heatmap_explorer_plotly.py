#%%
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
from ipywidgets import Output, VBox
#pio.renderers.default = "browser"

#def heatmap_viewer_plotly(array, title = "2d"):
array = np.random.random((200,100))

hline=0
vline=0

shape = np.shape(array)
x = np.arange(shape[0])
y = np.arange(shape[1])
#out = Output()
#@out.capture(clear_output=True)
def update_spikes(trace, point, selector):
    input()
    print("click")

fig = make_subplots(2, 2, 
    #shared_xaxes=True, shared_yaxes=True,
    specs = [[{},{}],[{}, None]],
    vertical_spacing = 0,
    )

heatmap=go.Heatmap(x=x, y=y, z=array.T, coloraxis = 'coloraxis')
#heatmap.on_click(update_spikes)
fig.add_trace(heatmap, row=1, col=1)






fig['layout']['yaxis']['scaleanchor']='x'
fig['layout']['coloraxis']['colorbar']['x']=0.5
fig['layout']['coloraxis']['colorbar']['outlinecolor']='black'
fig['layout']['coloraxis']['colorbar']['outlinewidth']=1
fig['layout']['coloraxis']['colorbar']['y']=0.56
fig['layout']['coloraxis']['colorbar']['len']=0.45
fig['layout']['coloraxis']['colorbar']['yanchor']='bottom'
fig['layout']['coloraxis']['colorbar']['thickness']=20
fig['layout']['coloraxis']['colorbar']['ticks']='outside'
fig['layout']['coloraxis']['colorscale']='RdBu'

fig['layout']['hovermode']='x'


fig['layout']['xaxis']['showspikes']=True
fig['layout']['yaxis']['showspikes']=True
fig['layout']['xaxis']['spikemode']='across'
fig['layout']['yaxis']['spikemode']='across'

#fig['layout']['xaxis']['showgrid']=False
fig['layout']['xaxis']['constrain']='domain'
#fig['layout']['yaxis']['showgrid']=False
fig['layout']['yaxis']['constrain']='domain'

#fig['layout']['xaxis']['tickmode']='array'
#fig['layout']['xaxis']['tickvals']=x

#fig['layout']['yaxis']['tickmode']='array'
#fig['layout']['yaxis']['tickvals']=y

fig.update_layout(height=600, width=800)

fig.update_xaxes(showline=True, linewidth=0.7, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=0.7, linecolor='black', mirror=True)

    #fig['layout']['grid']['yside'] = 'left plot'

    #return fig

#%%
#array = np.random.random((200,100))
#heatmap_viewer_plotly(array)
#%%