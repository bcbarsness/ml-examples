import quandl, math, datetime, config, time, pickle, os
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


'''
Predicts the adjusted close value of Google stock 33 days in the future using
LinearRegression (or commented SVM) & quandl data to train/test. This feature
set is not a good indicator/predictor but does successfully exercise the alg.
'''

style.use('ggplot')

quandl.ApiConfig.api_key = config.quandl_api_key
df = quandl.get("WIKI/GOOGL")

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
print("Predict Adjusted Closing Price {} days in the future".format(forecast_out))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Use the pickle file if it exists, otherwise train and dump pickle
picklefilename = 'stockpriceprediction.pickle'
if os.path.isfile('./' + picklefilename):
    pickle_in = open(picklefilename,'rb')
    clf = pickle.load(pickle_in)
else:
    clf = LinearRegression(n_jobs=-1)
    #clf = svm.SVR(kernel='linear')

    start = time.time()
    clf.fit(X_train, y_train)
    print("Training Time: ", time.time() - start)

    with open(picklefilename,'wb') as f:
        pickle.dump(clf, f)

accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
