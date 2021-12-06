import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

st.title("Researching about Amazon, Tesla Stocks")

# Numeric columns in df1
df1 = pd.read_csv("C:\\Amazon.csv")
df1 = df1.applymap(lambda x: np.nan if x == " " else x)
def can_be_numeric(c):
    try:
        pd.to_numeric(df1[c])
        return True
    except: 
        return False
good_cols_a = [c for c in df1.columns if can_be_numeric(c)]
df1[good_cols_a] = df1[good_cols_a].apply(pd.to_numeric, axis = 0)

# Numeric columns in df2
df2 = pd.read_csv("C:\\TSLA.csv")        
df2 = df2.applymap(lambda x: np.nan if x == " " else x)
def can_be_numeric(d):
    try:
        pd.to_numeric(df2[d])
        return True
    except:
        return False
good_cols_t = [d for d in df2.columns if can_be_numeric(d)]
df2[good_cols_t] = df2[good_cols_t].apply(pd.to_numeric, axis = 0)

# Choose the Dashboard
option = st.sidebar.selectbox("Choose the Dashboard", 
                              ("Each type of Stock","Stocks Comparision","Stock Price Prediction", "References"))
st.header(option)

# Each type of Stock #
if option == "Each type of Stock":
    symbol = st.sidebar.selectbox("Choose the Stock", ("AMZN","TSLA"))
    
    # AMZN #
    if symbol == "AMZN":
        st.subheader("AMZN Basic Info")
        amzn_data = yf.Ticker("amzn")
        amzn_logo = '<img src=%s>' %amzn_data.info['logo_url']
        st.markdown(amzn_logo, unsafe_allow_html=True)
        amzn_summary = amzn_data.info["longBusinessSummary"]
        st.info(amzn_summary)
        
        st.subheader("AMZN 10 Highest Trading Days")
        amzn_his = df1.nlargest(10,["Volume"])
        st.write(amzn_his)
        st.markdown("From the data, we see that *the volume trading of AMZN is higher in the past*, specifically in 1997, 1998, 1999, 2006, and 2007.")
        st.markdown("The reason was because of **the stock market crashes happening in these years**. In 1997, a mini-crash is a global stock market crash that was caused by an economic crisis in Asia. In 1998-1999, the collapse of hedge fund Long Term Capital Management rattled the markets. And Great Recession happened 2006-2007 leading to the crash of market in 2008.")
        
        st.subheader("AMZN Stock Price Chart")
        my_chart1 = alt.Chart(df1).mark_point().encode(
            x = st.selectbox("Choose your x-value:", good_cols_a),
            y = st.selectbox("Choose your y-value:", good_cols_a),
            color = alt.Color("Volume", scale=alt.Scale(scheme='goldred')),
            size = "High",
            tooltip = ["Date","Open","Close"])
        st.altair_chart(my_chart1)
        st.markdown("Observing the *'Adj Close'* axis and *'Volume'* axis, we can see that *AMZN is not traded a lot recently*. It might be because of the *expensive price of an AMZN's share*. Up to 2021, a share of AMZN has costed around *$3000* to purchase.")        
    
    # TSLA #
    if symbol == "TSLA":
        st.subheader("TSLA Basic Info")
        tsla_data = yf.Ticker("tsla")
        tsla_logo = '<img src=%s>' %tsla_data.info['logo_url']
        st.markdown(tsla_logo, unsafe_allow_html=True)
        tsla_summary = tsla_data.info["longBusinessSummary"]
        st.info(tsla_summary)
        
        st.subheader("TSLA 10 Highest Trading Days")
        tsla_his = df2.nlargest(10,"Volume")
        st.write(tsla_his)
        st.markdown("Unlike AMZN, TSLA has had a really *high trading volume in these recent years*. It can be up to *300 millions* trades in a day, while the highest of AMZN is only 100 millions.")
        
        st.subheader("TSLA Stock Price Chart")
        my_chart2 = alt.Chart(df2).mark_point().encode(
            x = st.selectbox("Choose your x-value:", good_cols_t),
            y = st.selectbox("Choose your y-value:", good_cols_t),
            color = alt.Color("Volume", scale=alt.Scale(scheme='teals')),
            size = "High",
            tooltip = ["Date","Open","Close"])
        st.altair_chart(my_chart2)
        st.markdown("Choosing the *'Volume'* and *'Adj Close'* axes, we can tell **how successfully Tesla stock has been in these recent years**. Tesla, in 2020, was *up an incredible 695%*. The reason was because of the *EV consumption* coming from China in 2020, and Europe in 2021. In the first 9 months of 2020, *Tesla Revenues in China grew over 90%*.")

# Stock Comparision #
# Comapring the stocks chart from our datasets #
if option == "Stocks Comparision":
    st.subheader("Comparing ['AMZN', 'TSLA'] from Our Datasets")
    my_chart3 = alt.Chart(df1).mark_point().encode(
            x = "Adj Close",
            y = "Volume",
            color = alt.value('blue'),
            opacity = "Low",
            size = "High",
            tooltip = ["Date","Open","Close"])
    my_chart4 = alt.Chart(df2).mark_square().encode(
            x = "Adj Close",
            y = "Volume",
            color = alt.value('orange'),
            opacity = "Low",
            size = "High",
            tooltip = ["Date","Open","Close"])
    st.altair_chart(my_chart3 + my_chart4)
    st.markdown("We can clearly tell the major difference between AMZN and TSLA when observing *'Adj Close'* and *'Volume'* axes. While AMZN has **higher price for a share**, TSLA has **higher volume trading**.")

# Comparing table in yfinance #
    st.subheader("Comparing ['AMZN', 'TSLA'] Table from yfinance Library")
    tickers = st.multiselect("Pick your stock(s)", ("AMZN", "TSLA"))
    dataframe = pd.DataFrame()
    for ticker in tickers:
        var = yf.Ticker(ticker).info
        frame = pd.DataFrame([var])
        dataframe = dataframe.append(frame)
    v = st.multiselect("Choose your value(s)", dataframe.columns)
    st.table(dataframe[v])

# Comparing stocks chart in yfinance #    
    st.subheader("Comparing {} Chart from yfinance Library".format(tickers))
    start_date = st.date_input("Start Date", datetime.date(2010,1,1))
    end_date = st.date_input("End Date", datetime.date(2021,11,1))
    def relativeret(df):
        rel = df.pct_change()
        cumret = (1+rel).cumprod()-1
        cumret = cumret.fillna(0)
        return cumret
    if len(tickers) > 0:
        chart_option = st.selectbox("Choose your compared value:", ("Volume","High", "Low", "Adj Close"))
        if chart_option == "Adj Close":
            df = relativeret(yf.download(tickers, start_date, end_date)[chart_option])
        else:
            df = yf.download(tickers,start_date, end_date)[chart_option]
        st.line_chart(df)
    st.markdown("When we choose to compare *'Volume'*, *'High'*, or *'Low'* values. The chart from yfinance looks **identical** to the previous Altair Chart from our datasets. The only differencce is when choosing *'Adj Close'* value of the two stocks. Instead of comparing their adjust actual closing prices, I learn a technique from *Algovibes Channel* on *YouTube* to **convert the prices into cumulative returns**. (Cumulative return is the total change in the investment price over a set time—an aggregate return, not an annualized one). It helps us easier to identify if these stocks are worth investing as well as easier to compare them without looking at each of the volume, high, low, or open, close charts.")
    st.markdown("Moreover, *accessing and analyzing datas from yfinance library* are much better and faster than *downloading and reading the datasets* when we want to build a comparing stocks app. In yfinance library, we can simply pick any stock and compare them all at once. However, things are harder if we choose to download each of the dataset seperately and then import them into Python. In fact, I only know about yfinance library after choosing the two 'AMZN' and 'TSLA' datasets and started to work on them. That is when I realize how awesome yfinance library is. With our datasets, we can only *abserve the 'Open', 'Close', 'High', 'Low', 'Adj Close', and 'Volume'* values, but yfinance library could **provide us any information about one specific stock** that we wish to know.")            

    
# Stock Price Prediction #
if option == "Stock Price Prediction":
    # Predict TSLA price by Linear Regression #
    st.subheader("Predict TSLA Closing Price by Linear Regression")
    st.markdown("In this Linear Regression Model, I has chosen *X* to be the *'Open', 'High', 'Low', and 'Volume'* columns in TSLA dataset in order to predict the value of *y - 'Close'*.   ")
    
    X1 = df2[["High", "Low", "Open", "Volume"]].values
    y1 = df2["Close"].values
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2,random_state=0)
    regression = LinearRegression()
    regression.fit(X1_train, y1_train)
    predict = regression.predict(X1_test)
    df2_sub = pd.DataFrame({"Actual": y1_test.flatten(), "Predict": predict.flatten()})
    rows = st.multiselect("Choose row(s):", df2_sub.index)
    selected_rows = df2_sub.loc[rows]
    st.write("Actual and Predicted Closing Price of AMZN Choosing from Row(s)", selected_rows)
    show_error = st.checkbox("Show Mean Absolute Error and Mean Square Error")
    if show_error:
        st.write("Mean Absolute Error:", metrics.mean_absolute_error(y1_test, predict))
        st.write("Mean Squared Error:", metrics.mean_squared_error(y1_test, predict))   
    
    fig, ax = plt.subplots()
    x_axis = df2_sub.tail(60).index
    y_axis = df2_sub.tail(60)["Actual"]
    x_value = df2_sub.tail(60).index
    y_value = df2_sub.tail(60)["Predict"]

    ax.plot(x_axis, y_axis, color='r', label="Actual")
    ax.plot(x_value, y_value, color='c', label="Predicted")
    plt.legend(loc='upper left')
    plt.title("Actual and Predicted Closing Price of TSLA by Linear Regression")
    st.pyplot(fig)
    
    st.markdown("In this Linear Regression Model, the *Predicted Price was really close to the Actual Price*. However, we **cannot predict stocks future by this model since stocks' price are not a linear relationship**. Stock market could be affected by many factors like political upheaval, interest rates, current events, exchange rate fluctuations, natural calamities. Therefore, **it is really hard for a machine to tell the future of a stock even if only predicts a day ahead**.")
    
    # Predict TSLA price by Neutral Network #
    st.subheader("Predict TSLA Closing Price by Neutral Network")
    st.markdown("Leanring more about *Recurrent Neutral Network (RNN)* through the *KGP Talkie Channel* on *YouTube*, I will try to build and train a neural network *predicting TSLA closing price*.")
    st.markdown("Taken *X-train as 'Open','High', 'Low', and 'Volume' of TSLA until 01-01-2020*, I wanted to compute the value of *y-predict TSLA in 60 days from 2021-07-22 to 2021-10-14*.")
    st.markdown("To do this, first I needed to *prepare the data* for X-train and y-train. Then, I *built the neutral network model* to train X-train and y-train. Next, I made X-test and y-test, and *computed y-predict by taking* **model(X-test)**. Finally, I *graphed the chart to compare the values* of y-test (Actual Value of TSLA) and y-predict(Predicted Value of TSLA). ")
    
    df2_training = df2[df2["Date"] < "2020-01-01"].copy()
    df2_test = df2[df2["Date"] >= "2020-01-01"].copy()
    df2_training = df2_training.drop(["Date", "Adj Close"], axis=1)
    
    scaler = MinMaxScaler()
    df2_training = scaler.fit_transform(df2_training)

    X_train = []
    y_train = []

    for i in range(60, df2_training.shape[0]):
        X_train.append(df2_training[i-60:i])
        y_train.append(df2_training[i,0])
    
    X_train, y_train = (np.array(X_train), np.array(y_train))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    
    model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (60,5)), 
        keras.layers.Flatten(), 
        keras.layers.Dense(40, activation="relu"), 
        keras.layers.Dense(190, activation="relu"), 
        keras.layers.Dense(1)  
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"],)
    history = model.fit(X_train, y_train,epochs=50, validation_split=0.2) 
    
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"])
    ax.plot(history.history["val_loss"])
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.legend(["train","validation"],loc="upper right")
    plt.title("The Plot of the Training Loss and Validation Loss")
    st.pyplot(fig)
    
    st.markdown("I had a plot of the Training Loss and Validation Loss in here. It was *much better* than I expected since *my accuracies were so small*, only about 0.15 to 0.25. In the *YouTube Video*, the instruction was about **building a popular RNN called the Long Short-Term Model Network (LSTM)**. However, I kept getting the error *NotImplementedError: Cannot convert a symbolic Tensor (2nd_target:0) to a numpy array* in the Input Layer. I tried to search for solutions but many said that **the version of NumPy and TensorFlow caused that** so I gave up after a few attemepts of fixing.")
    
    df2_training = df2[df2["Date"] < "2020-01-01"].copy()
    past_60 = df2_training.tail(60)
    dfn = past_60.append(df2_test)
    dfn = dfn.drop(["Date","Adj Close"],axis=1)
    inputs = scaler.transform(dfn)
    
    X_test = []
    y_test = []
    
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])
        y_test.append(inputs[i,0])
            
    X_test, y_test = (np.array(X_test), np.array(y_test))
    
    y_pred = model.predict(X_test)
    scale = 1/1.19371628e-02 #1.19371628e-02 was from scaler.scale_#
    y_pred = y_pred*scale
    y_test = y_test*scale
    
    fig3, ax3 = plt.subplots()
    ax3.plot(y_test, color='r', label="Actual TSLA Closing Price")
    ax3.plot(y_pred, color='c', label="Predicted TSLA Closing Price")
    plt.legend(loc='upper left')
    plt.title("Actual and Predicted Closing Price of TSLA by Neutral Network")
    st.pyplot(fig3)
    
    st.markdown("This is my graph visualizing the **Actual and Prediced TSLA Price**. I think my model work just fine in order to create a line that looks pretty identical to the real-time data.")
    

# References #
if option == "References":
    code1 = '''start_date = st.date_input("Start Date", datetime.date(2010,1,1))
                end_date = st.date_input("End Date", datetime.date(2021,11,1))
                def relativeret(df):
                    rel = df.pct_change()
                    cumret = (1+rel).cumprod()-1
                    cumret = cumret.fillna(0)
                    return cumret'''
    st.code(code1, language='python')
    link1 = "[Algovibes](https://www.youtube.com/watch?v=Km2KDo6tFpQ)"
    st.write("This portion of the app was taken from ", link1)
    
    
    code2 = '''tickers = st.multiselect("Pick your stock(s)", ("AMZN", "TSLA"))
                dataframe = pd.DataFrame()
                for ticker in tickers:
                    var = yf.Ticker(ticker).info
                    frame = pd.DataFrame([var])
                    dataframe = dataframe.append(frame)
                v = st.multiselect("Choose your value(s)", dataframe.columns)
                st.table(dataframe[v]) '''
    st.code(code2, language='python')
    link2 = "[Algovibes](https://www.youtube.com/watch?v=YsnPlQyCYfo)"
    st.write("This portion of the app was taken from ", link2)
    
    
    code3 = '''X1 = df2[["High", "Low", "Open", "Volume"]].values
               y1 = df2["Close"].values
               X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2,random_state=0)
               regression = LinearRegression()
               regression.fit(X1_train, y1_train)
               predict = regression.predict(X1_test)
               df2_sub = pd.DataFrame({"Actual": y1_test.flatten(), "Predict": predict.flatten()})'''
    st.code(code3, language='python')
    link3 = "[Dhanashri Kolekar](https://www.youtube.com/watch?v=uEGZ68NH-sM)"
    st.write("This portion of the app was taken from ", link3)
    
    
    code4 = ''' df2_training = df2[df2["Date"] < "2020-01-01"].copy()
        df2_test = df2[df2["Date"] >= "2020-01-01"].copy()
        df2_training = df2_training.drop(["Date", "Adj Close"], axis=1)
    
        scaler = MinMaxScaler()
        df2_training = scaler.fit_transform(df2_training)

        X_train = []
        y_train = []

        for i in range(60, df2_training.shape[0]):
            X_train.append(df2_training[i-60:i])
            y_train.append(df2_training[i,0])
    
        X_train, y_train = (np.array(X_train), np.array(y_train))
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))'''
    st.code(code4, language='python')
    link4 = "[KGP Talkie](https://www.youtube.com/watch?v=arydWPLDnEc&t=1677s)"
    st.write("This portion of the app was taken from", link4)
        
    
    
    code5 = ''' df2_training = df2[df2["Date"] < "2020-01-01"].copy()
                past_60 = df2_training.tail(60)
                dfn = past_60.append(df2_test)
                dfn = dfn.drop(["Date","Adj Close"],axis=1)
                inputs = scaler.transform(dfn)
    
                X_test = []
                y_test = []
    
                for i in range(60, inputs.shape[0]):
                    X_test.append(inputs[i-60:i])
                    y_test.append(inputs[i,0])
            
                X_test, y_test = (np.array(X_test), np.array(y_test))
    
                y_pred = model.predict(X_test)
                scale = 1/1.19371628e-02 #1.19371628e-02 was from scaler.scale_#
                y_pred = y_pred*scale
                y_test = y_test*scale'''
    st.code(code5, language='python')
    st.write("This portion of the app was taken from", link4)
    
    
    