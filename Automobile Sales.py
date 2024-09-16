import dash
from dash import html,dcc
from dash.dependencies import Input,Output,State
import pandas as pd
import plotly.express as px


df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DV0101EN-SkillsNetwork/Data%20Files/historical_automobile_sales.csv")
year_list = [i for i in range(1980, 2024, 1)]


app=dash.Dash()
app.config.suppress_callback_exceptions = True

app.layout=html.Div(children=[html.H1('Automobile Sales Statistics Dashboard',style={'textAlign':'center',
                                      'color':'#503D36','fontsize':24}),
                              html.Div([dcc.Dropdown(id='dropdown-statistics',
                                                     options=[{'label':'Yearly Statistics','value':'Yearly Statistics'},
                                                              {'label':'Recession Period Statistics','value':'Recession Period Statistics'}],
                                                              placeholder='Select a report type',value='Select Statistics',
                                                              style={'width':'80%','padding':'3px','fontsize':'25px','text-align-last':'center'}),
                                        dcc.Dropdown(id='select-year',
                                                     options=[{'label':i,'value':i}for i in year_list],placeholder='Select Year',
                                                     style={'width':'80%','padding':'3px','fontsize':'25px','text-align-last':'center'} )
                                        ]), 
                              html.Div(id='output-container', className='chart-grid'),                               
                               ])
 
@app.callback(Output('isDisabled','disabled'),
              Input(component_id='dropdown-statistics',component_property='value'))

def update_input_container(Input_report):
    if Input_report=='Yearly Statistics':
        return False
    else:
        return True

@app.callback(Output(component_id='output-container',component_property='children'),
              [Input(component_id='select-year',component_property='value'),
               Input(component_id='dropdown-statistics',component_property='value')])

def reg_chart(input_year,input_type):
    if input_type == 'Recession Period Statistics':
        r_df = df[df['Recession'] == 1]

        yearly_rec=r_df.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart1 = dcc.Graph(figure=px.line(yearly_rec, x='Year',y='Automobile_Sales',
                                            title='Average Automobile sales')) 
        
        rv_df=r_df.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        R_chart2 = dcc.Graph(figure=px.bar(rv_df, x='Vehicle_Type',y='Automobile_Sales',
                                            title='Average number of vehicles sold by vehicle type'))      
           
        re_df= r_df.groupby('Vehicle_Type')['Advertising_Expenditure'].sum()
        R_chart3 = dcc.Graph(figure=px.pie(re_df,values=re_df,names=re_df.index,labels=re_df.index,
                                           title='Total expenditure share by vehicle type'))

        R_chart4 = dcc.Graph(figure=px.bar(r_df,x='unemployment_rate',y='Automobile_Sales',color='Vehicle_Type',  
                                            title='Effect of unemployment rate on vehicle type and sales'))      
        
        return [html.Div(className='chart-item', children=[html.Div(children=R_chart1),html.Div(children=R_chart2)],
                        style={'display': 'flex'}),
                html.Div(className='chart-item', children=[html.Div(children=R_chart3),html.Div(children=R_chart4)],
                        style={'display': 'flex'})
               ]
    elif(input_type=='Yearly Statistics'):

        y_df=df.groupby('Year')['Automobile_Sales'].mean().reset_index()
        Y_chart1 = dcc.Graph(figure=px.line(y_df, x='Year',y='Automobile_Sales',
                title="Automobile sales fluctuate over Recession Period"))
        
        m_df=df[df['Year']==input_year]

        Y_chart2 = dcc.Graph(figure=px.line(m_df, x='Month',y='Automobile_Sales',
                title='Total Monthly Automobile sales in {}'.format(input_year)))
        
        v_df=m_df.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        Y_chart3 = dcc.Graph(figure=px.bar(v_df, x='Vehicle_Type',y='Automobile_Sales',
                title='Total Monthly Automobile sales in {}'.format(input_year)))       

        e_df=m_df.groupby('Vehicle_Type')['Advertising_Expenditure'].sum()
        Y_chart4 = dcc.Graph(figure=px.pie(values=e_df,names=e_df.index,labels=e_df.index,
                title='Total Advertisement Expenditure for each vehicle in {}'.format(input_year)))  
        
        return [html.Div(className='chart-item', children=[html.Div(children=Y_chart1),html.Div(children=Y_chart2)],
                        style={'display': 'flex'}),
               html.Div(className='chart-item', children=[html.Div(children=Y_chart3),html.Div(children=Y_chart4)],
                     style={'display': 'flex'})
               ]
 
if __name__=='__main__':
    app.run_server(port=8055)
