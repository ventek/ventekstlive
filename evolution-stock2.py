'''
This script is used to build a stock market dashboard to show charts and data of stock
'''

#Import the libraries
import asyncio
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image
import pandas as pd
from datetime import timedelta
import datetime
import pytz
import numpy as np
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
import mplfinance as mpf
import matplotlib.animation as animation
import webbrowser
import warnings

FFwriter = animation.FFMpegWriter()

# Writer = animation.writers['pillow']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# Writer = animation.FFMpegWriter(fps=30, codec='libx264') # Or
# Writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)

st.set_page_config(layout="wide")
#Add title and image
st.write('''
# Evolution ST: Stock Market Analysis
by Ventek' evolution 
''')


# image = Image.open('C:/Users/zane/Documents/003-VentekTechnologies/002-truelancer/002-TL0001/web-app/new-cover-4.png')
# st.image(image, use_column_width=True)

#Create Sidebar Header
url = 'https://www.venteksa.com'
url2 = 'https://www.eulatemplate.com/live.php?token=rz4OIb3q4kAHvPrAVEB7NFyfMukqQDCB'
if st.button('Ventek Website'):
    webbrowser.open_new_tab(url)
if st.button('EULA Agreement'):
    webbrowser.open_new_tab(url2)



# st.sidebar.header('User Input')

###################################################################################################################




warnings.filterwarnings("ignore", category=RuntimeWarning)

st.header('Live Data\n')

apik = st.text_input('API KEY', 'Insert Key')
idsec = st.text_input('API SEC ID', 'Insert ID')


api = tradeapi.REST(apik,
                    idsec,
                    'https://paper-api.alpaca.markets')
dbDateFormat = "%Y-%m-%d %H:%M:%S"
# pd.set_option('mode.chained_assignment', 'raise')
# pd.options.mode.chained_assignment = None  # default='warn'
def get_data(symbol, lookback):

    # Current time in UTC
    now_est = datetime.datetime.now(pytz.timezone('EST'))

    # Dates between which we need historical data
    from_date = (now_est - timedelta(days=lookback)).strftime(dbDateFormat)
    to_date = now_est.strftime(dbDateFormat)

    # returns open, high, low, close, volume, vwap
    all_data = api.polygon.historic_agg_v2(symbol, 5, 'minute', _from=from_date, to=to_date).df
    all_data.drop(columns=['volume'], inplace=True)
    all_data.drop(columns=['vwap'], inplace=True)
    all_data.replace(0, method='bfill', inplace=True)
    all_data.index.name = "Date"
    all_data.index = pd.to_datetime(all_data.index)
    all_data['Date'] = all_data.index
    # all_data = all_data.reset_index(level=0, drop=True).reset_index()
    return all_data

data = get_data('CAT', 4)

df = data

def reg_calc(df, ival):
    x = list(range(0, len(data['high'].iloc[ival:(20 + ival)])))
    y_h = data['high'].iloc[ival:(20 + ival)]
    y_l = data['low'].iloc[ival:(20 + ival)]
    fit_h = np.polyfit(x, y_h, 1)
    fit_fn_h = np.poly1d(fit_h)
    fit_l = np.polyfit(x, y_l, 1)
    fit_fn_l = np.poly1d(fit_l)
    df.insert(5, 'Hreg', fit_fn_h(x))
    df.insert(6, 'Lreg', fit_fn_l(x))

    # if 'Hreg' in df.columns:
    #     return df
    # else:
    #     df.insert(5,'Hreg', fit_fn_h(x))
    #     df.insert(6, 'Lreg', fit_fn_l(x))
    #     return df
    return df


def buy_sell_updowntrend(df):
    Buy = []
    Sell = []
    flag = -1
    for i in range(0, len(df)):
        if df['Lreg'][i] < df['close'][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(df['close'][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif df['Hreg'][i] > df['close'][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(df['close'][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    return (Buy, Sell)


pkwargs=dict(type='candle', tz_localize=False)
plt.style.use('fivethirtyeight')

fig, axes = mpf.plot(data.iloc[0:20],returnfig=True,volume=False,
                     figsize=(11,8),
                     title='\n\nLive Data',
                     **pkwargs)

ax_main = axes[0]
ax_emav = ax_main


def animate(ival):

    if (20+ival) > len(df):
        print('no more data to plot')
        ani.event_source.interval *= 3
        if ani.event_source.interval > 600000:
            return
        return


    data = df.iloc[ival:(20 + ival)]
    data_c = data.copy()
    df_reg = reg_calc(data_c, ival)
    Buy, Sell = buy_sell_updowntrend(df_reg)
    df_reg['Signal_Buy'] = Buy
    df_reg['Signal_Sll'] = Sell
    df_reg_c = df_reg


    reg_plot = [
        mpf.make_addplot(df_reg_c['Hreg'], ax=ax_emav, type='line', color='g'),
        mpf.make_addplot(df_reg_c['Lreg'], ax=ax_emav, type='line', color='r'),
        mpf.make_addplot(df_reg_c['Signal_Buy'], ax=ax_emav, type='scatter',markersize=200,marker='^', color='b'),
        mpf.make_addplot(df_reg_c['Signal_Sll'], ax=ax_emav, type='scatter',markersize=200,marker='v',color='black'),
            ]
    # datepairs = [(d1, d2) for d1, d2 in zip(dates, dates[1:])]
    d1 = df.index[0]
    d2 = df.index[-1]
    datepairs = [(d1, d2)]
    ax_main.clear()
    ax_emav.clear()
    mc = mpf.make_marketcolors(up='g', down='r')
    s = mpf.make_mpf_style(marketcolors=mc)
    # mpf.plot(data ,ax=ax_main, tlines=[dict(tlines=datepairs, tline_use='high',tline_method='least-squares', colors='g'),
    #                                    dict(tlines=datepairs, tline_use='low',tline_method='least-squares', colors='b')], addplot=reg_plot, **pkwargs)
    mpf.plot(data, ax=ax_main, addplot=reg_plot, style=s, **pkwargs)

# Writer = animation.FFMpegWriter(fps=30, codec='libx264') # Or

# Writer = animation.FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, animate, blit=False, interval=60000)


async def watch(test):
    with open("myvideo.html", "w") as f:
        print(ani.to_html5_video(), file=f)
    r = await asyncio.sleep(1)
    # HtmlFile = line_ani.to_html5_video()
    HtmlFile = open("myvideo.html", "r")
    # HtmlFile="myvideo.html"
    source_code = HtmlFile.read()
    components.html(source_code, height=900, width=1200)


test = st.empty()
asyncio.run(watch(test))


# if st.button("Click me."):
#     st.image("https://cdn11.bigcommerce.com/s-7va6f0fjxr/images/stencil/1280x1280/products/40655/56894/Jdm-Decals-Like-A-Boss-Meme-Jdm-Decal-Sticker-Vinyl-Decal-Sticker__31547.1506197439.jpg?c=2", width=200)











