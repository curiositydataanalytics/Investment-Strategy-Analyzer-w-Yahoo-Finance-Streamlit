# Data manipulation
import numpy as np
import datetime as dt
import pandas as pd
import geopandas as gpd

# Database and file handling
import os

# Data visualization
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk

import yfinance as yf
import financedatabase as fd
from streamlit_extras.colored_header import ST_COLOR_PALETTE

path_cda = '\\CuriosityDataAnalytics'
path_wd = path_cda + '\\wd'
path_data = path_wd + '\\data'

# App config
#----------------------------------------------------------------------------------------------------------------------------------#
# Page config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .element-container {
        margin-top: -5px;
        margin-bottom: -5px;
        margin-left: -5px;
        margin-right: -5px;
    }
    img[data-testid="stLogo"] {
                height: 6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# App title
st.title("Investment Strategy Analyzer")


#
#


@st.cache_data
def load_data():
    ticker_list = pd.concat([fd.ETFs().select().reset_index().query("category_group == 'Equities'")[['symbol', 'name']],
                             fd.Equities().select().reset_index().query("sector == 'Information Technology'")[['symbol', 'name']]])
    ticker_list = ticker_list[(ticker_list.symbol.notna()) & (ticker_list.name.notna())]
    ticker_list['symbol_name'] = ticker_list.symbol + ' - ' + ticker_list.name
    ticker_list = ticker_list[ticker_list.symbol_name.str.startswith('^') == False].copy()

    return ticker_list
ticker_list = load_data()


with st.expander('Portfolio'):
    pf = pd.DataFrame({'Asset' : pd.Series(dtype='str'),
                       'Label' : pd.Series(dtype='str'),
                        'Amount' : pd.Series(dtype='float'),
                        'Frequency' : pd.Series(dtype='str'),
                        'Start Date' : pd.Series(dtype='object'),
                        'End Date' : pd.Series(dtype='object')})
    pf = st.data_editor(pf,
                        hide_index=True,
                        num_rows='dynamic',
                        column_config={'Asset': st.column_config.SelectboxColumn(width='large', options=ticker_list.symbol_name.unique(), required=True),
                                        'Amount': st.column_config.NumberColumn(width='small', min_value=0.0, format="$%.2f", step=1, required=True),
                                        'Frequency': st.column_config.SelectboxColumn(options=['Daily', 'Weekly', 'Biweekly', 'Monthly', 'Quarterly', 'Yearly'], required=True),
                                        'Start Date': st.column_config.DateColumn(width='small',required=True),
                                        'End Date': st.column_config.DateColumn(width='small', required=True)},
                        use_container_width=True)


if not pf.empty:
    # Extract yfinance data
    dfseries = []
    for asset in pf.Asset.str.split(' - ', expand=True)[0].unique():
        for label in pf[pf.Asset.str.split(' - ', expand=True)[0] == asset].Label.unique():
            df = yf.Ticker(asset).history(period='11y', interval='1d')
            df['Symbol'] = asset
            df['Label'] = label
            dfseries.append(df)
    dfseries = pd.concat(dfseries)
    dfseries = (dfseries
        .groupby(['Symbol', 'Label'], group_keys=False)
        .apply(lambda g: g.reindex(pd.date_range(start=g.index.min(), end=g.index.max())))
        .assign(Symbol=lambda g: g['Symbol'].ffill(),
                Label=lambda g: g['Label'].ffill(),
                Open=lambda g: g['Open'].ffill(),
                High=lambda g: g['High'].ffill(),
                Low=lambda g: g['Low'].ffill(),
                Close=lambda g: g['Close'].ffill())
    )
    dfseries['StartDate'] = pd.to_datetime(dfseries.index.date)
    dfseries['WeekStartDate'] = (dfseries['StartDate'] - pd.to_timedelta((dfseries['StartDate'].dt.dayofweek) % 7, unit='D')).dt.date
    dfseries['MonthStartDate'] = dfseries['StartDate'].dt.to_period('M').dt.to_timestamp().dt.date
    dfseries['MidYearStartDate'] = dfseries['StartDate'].apply(
        lambda x: pd.Timestamp(f'{x.year}-01-01').date() if x.month <= 6 else pd.Timestamp(f'{x.year}-07-01').date()
    )
    dfseries['YearStartDate'] = dfseries['StartDate'].apply(lambda x: pd.Timestamp(f'{x.year}-01-01').date())
    
    # Build investment strategy df
    pf_long = []
    for i, grp in pf.groupby(['Asset', 'Label']):
        fcy = {'Daily': 'D', 'Weekly': 'W-MON', 'Biweekly': '2W-MON', 'Monthly': 'M', 'Quarterly': 'Q', 'Yearly': 'Y'}
        dt_range = pd.date_range(start=grp['Start Date'].iloc[0], end=grp['End Date'].iloc[0], freq=fcy[grp['Frequency'].iloc[0]])
        df = pd.DataFrame({
            'Asset': [grp['Asset'].iloc[0]] * len(dt_range),
            'Label': [grp['Label'].iloc[0]] * len(dt_range),
            'Amount': [grp['Amount'].iloc[0]] * len(dt_range),
            'Date': dt_range
        })
        pf_long.append(df)
    pf_long = pd.concat(pf_long, ignore_index=True)
    pf_long = pd.merge(pf_long, ticker_list[['symbol_name', 'symbol']], left_on='Asset', right_on='symbol_name', how='left')
    
    # Merge with yfinance data
    dfseries = pd.merge(dfseries, pf_long, left_on=['Symbol', 'Label', 'StartDate'], right_on=['symbol', 'Label', 'Date'], how='left')
    dfseries = dfseries[(dfseries.StartDate >= dfseries['Date'].min()) & (dfseries.StartDate <= dfseries['Date'].max())].copy()
    dfseries['Amount'] = dfseries.groupby('Label')['Amount'].fillna(0)
    
    dfserieslist = []
    for i, grp in dfseries.groupby(['Symbol', 'Label']):
        grp['price_pct'] = grp['Close'].pct_change().cumsum()
        grp['growth_dollars'] = grp['Amount'].iloc[0]
        for i in range(1, len(grp)):
            grp['growth_dollars'].iloc[i] = (grp['growth_dollars'].iloc[i-1] + grp['Amount'].iloc[i]) * (1 + grp['Close'].pct_change().iloc[i])
        grp['growth_pct'] = (grp['growth_dollars'] / grp['Amount'].cumsum()) - 1

        grp['dividend_dollars'] = ((grp.growth_dollars / grp.Close) * grp.Dividends).fillna(0)
        grp['growth_dollars_w_dividends'] = grp.growth_dollars + grp.dividend_dollars
        grp['growth_pct_w_dividends'] = (grp.growth_dollars_w_dividends / grp.Amount.cumsum()) - 1
        dfserieslist.append(grp)
    dfseries = pd.concat(dfserieslist)

with st.sidebar:
    st.logo(path_cda + '\\logo.png', size='large')
    st.empty()

    st.subheader('Plot settings')
    metric = st.radio('Metric', ['Price', '% Price Change', '% Portfolio Growth', '$ Portfolio Growth'])
    time_interval = st.radio('Time Interval', ['1 day', '1 week', '1 month', '6 months', '1 year'])
    divid = st.toggle('Include Dividends', value=False)

if not pf.empty:
    cols = st.columns((0.7,0.3))

    cols[0].subheader('Plot')

    time_var = {
        '1 day': 'StartDate',
        '1 week': 'WeekStartDate',
        '1 month': 'MonthStartDate',
        '6 months': 'MidYearStartDate',
        '1 year': 'YearStartDate'
    }.get(time_interval, 'YearStartDate')

    if divid:
        y_var = 'Close' if metric == 'Price' else 'price_pct' if metric == '% Price Change' else 'growth_pct_w_dividends' if metric == '% Portfolio Growth' else 'growth_dollars_w_dividends'
    else:
        y_var = 'Close' if metric == 'Price' else 'price_pct' if metric == '% Price Change' else 'growth_pct' if metric == '% Portfolio Growth' else 'growth_dollars'
    
    fig = px.line(dfseries[dfseries.StartDate == dfseries[time_var]], x='StartDate', y=y_var, color='Label')
    color_mapping = {trace.name: trace.line.color for trace in fig.data if 'color' in trace.line}
    if divid:
        for j in dfseries.Label.unique():
            for i in dfseries[(dfseries['dividend_dollars'] > 0) & (dfseries['Label'] == j)].StartDate.unique():
                color = color_mapping.get(j, 'black')
                fig.add_vline(x=i, line=dict(color=color, width=2, dash='dash'))
                fig.update_layout(template="plotly_dark")
    cols[0].plotly_chart(fig)

    cols[1].subheader('Summary')
    y_var1 = 'growth_dollars' if divid else 'growth_dollars_w_dividends'
    y_var2 = 'growth_pct' if divid else 'growth_pct_w_dividends'
      
    fig = px.treemap(dfseries.groupby('Label')[y_var1].last().reset_index(), path=['Label'], values=y_var1)
    fig.update_traces(customdata=np.stack((dfseries.groupby('Label')[y_var2].last().reset_index()[y_var2]*100,
                                  dfseries.groupby('Label').Amount.sum().reset_index()['Amount']), axis=-1),
                      textinfo="label+value",
                      texttemplate="%{label}<br>$%{value:,.2f}<br>%{customdata[0]:.2f}% from $%{customdata[1]:,.2f}",
                      textfont=dict(size=20))
    cols[1].plotly_chart(fig)
    