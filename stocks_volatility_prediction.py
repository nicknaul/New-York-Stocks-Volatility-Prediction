from arch import arch_model
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import datetime
import yfinance as yahooFinance
import numpy as np
import plotly.express as px

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


ticker_symbol="AMZN"
def get_stocks(ticker):
    GetInformation = yahooFinance.Ticker(ticker)
    startDate = datetime.datetime(1999, 11, 5)
    endDate = pd.Timestamp.now()
    df= pd.DataFrame(GetInformation.history(start=startDate,end=endDate))
    df.sort_index(ascending=False, inplace=True)
    df.drop(columns=["Dividends"], inplace=True)
    df.drop(columns=["Stock Splits"], inplace=True)
    df=df.round(2)
    return df


def wrangle_data(ticker,n_observations):

    # Get table from database
    df=get_stocks(ticker=ticker)
    df=df.head(n_observations+1)
    # Sort DataFrame ascending by date
    df.sort_index(ascending=True, inplace=True)

    # Create "return" column
    df["Return"] = df["Close"].pct_change() *100

    # Return returns
    return df["Return"].dropna()



def forecast_(c,model):
       
    # Generate 10-day volatility forecast
    prediction = model.forecast(horizon=10, reindex=False).variance
    prediction_formatted = clean_prediction(prediction)
    return prediction_formatted
    

def clean_prediction(prediction):

    # Calculate forecast start date
    start = prediction.index[0] + pd.DateOffset(days=2)

    # Create date range
    prediction_dates = pd.bdate_range(start=start, periods=prediction.shape[1])

    # Create prediction index labels, ISO 8601 format
    prediction_index = [d.isoformat() for d in prediction_dates]

    # Extract predictions from DataFrame, get square root
    data = prediction.values.flatten() ** 0.5

    # Combine `data` and `prediction_index` into Series
    prediction_formatted = pd.Series(data, index=prediction_index)

    # Return Series as dictionary
    return prediction_formatted.to_dict()


def create_model(ticker):
    y_df= wrangle_data(ticker=ticker, n_observations=2500)
    df_daily_volatility = y_df.std()
    df_annual_volatility=df_daily_volatility**np.sqrt(252)
    cutoff_test = int(len(y_df) * 0.8)
    y_df_train = y_df.iloc[:cutoff_test]
    model = arch_model(
        y_df_train,
        p=1,
        q=1,
        rescale=False
    ).fit(disp=0)

    predictions = []

    # Calculate size of test data (20%)
    test_size = int(len(y_df) * 0.2)

    # Walk forward
    for i in range(test_size):
        # Create test data
        y_train = y_df.iloc[: -(test_size - i)]

        # Train model
        model = arch_model(y_train, p=1, q=1, rescale=False).fit(disp=0)

        # Generate next prediction (volatility, not variance)
        next_pred = model.forecast(horizon=1, reindex=False).variance.iloc[0, 0] ** 0.5

        # Append prediction to list
        predictions.append(next_pred)

    # Create Series from predictions list
    y_test_wfv = pd.Series(predictions, index=y_df.tail(test_size).index)

    g = forecast_(c=2, model=model)
    df =pd.DataFrame.from_dict(g, orient='index', columns=['Return'])
    df.index.name = 'Date'
    #print(df.head(10))
    #print(df.tail(10))
    fig = px.line(df, title= f'{ticker} Volatility Predictions for 10 days')

    fig.update_layout(
    plot_bgcolor="paleturquoise",
    paper_bgcolor="paleturquoise"
        )
    print("Current Returns (%):")
    print(y_df.tail(10).sort_index(ascending=False))
    return fig.show()


create_model(ticker_symbol)
