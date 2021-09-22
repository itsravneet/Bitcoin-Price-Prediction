import re
#import humanize
from IPython.display import clear_output, display
from pathlib import Path


from flask import Flask,request,jsonify, render_template,send_file
import io
import base64
import pickle
from bs4 import BeautifulSoup 
import requests
import lxml
from pickle import load
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import matplotlib.pyplot as plt 

app=Flask(__name__)
true=0
future=0
forecast_day=0
forecast_price=0
@app.route('/')
def home():
	return 	render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	global forecast_day
	global forecast_price
	#getting day forecast value from html toggle
	try:
		day = request.form['prediction']
	except:
		return render_template('index.html',prediction_text="Please choose a forecast day!")
	model=pickle.load(open("randomforestregressor_"+str(day)+".sav", 'rb'))


	#Scraping data
	source=requests.get('https://bitinfocharts.com/bitcoin/').text
	soup=BeautifulSoup(source,'lxml')
	c=''
	a=list(soup.find('span',itemprop="price").text.split(','))
	for i in a:
		c+=i
	c=float(c)
	priceUSD=c

	a,b=soup.find('td',id="tdid34").span.text.split(' ')
	c=float(a)
	median_transaction_fee=c

	c=''
	a=list(soup.find('td',id="tdid5").span.text.split(','))
	for i in a:
		c+=i
	c=c.split()[0]
	c=int(c)*priceUSD
	sentinusd=c

	a=soup.find('td',id="tdid15").abbr.text.split()
	if(a[1]=='T'):
		b=float(a[0])*(10**12)
	if(a[1]=='P'):
		b=float(a[0])*(10**15)
	difficulty=b
	a=soup.find('td',id="tdid16").abbr.text.split()
	if(a[1]=='T'):
		b=float(a[0])*(10**12)
	if(a[1]=='P'):
		b=float(a[0])*(10**15)
	if(a[1]=="E"):
		b=float(a[0])*(10**18)
	hashrate=b

	#predicting the future price
	test_point=np.array([median_transaction_fee,hashrate,sentinusd,difficulty,priceUSD])
	next_day_price=model.predict(test_point.reshape(1,-1))
	
	#scraping prices data
	chart_dict_list = [
                    {'url': 'https://bitinfocharts.com/comparison/bitcoin-median_transaction_fee.html', 'name': 'median_transaction_size'},
                    {'url': 'https://bitinfocharts.com/comparison/bitcoin-hashrate.html', 'name': 'hashrate'},
                    {'url': 'https://bitinfocharts.com/comparison/sentinusd-btc.html', 'name': 'send_usd'},
                    {'url': 'https://bitinfocharts.com/comparison/bitcoin-difficulty.html', 'name': 'difficulty'},
                    {'url': 'https://bitinfocharts.com/comparison/bitcoin-price.html', 'name': 'price'},
                    ]
	url = 'https://bitinfocharts.com'
	response = requests.get(url)
	soup = BeautifulSoup(response.text, 'html.parser')

	coin_dict_list = [{'coin': 'btc', 'full_name': 'bitcoin'}]
	for coin_dict in coin_dict_list:
	  coin_dict['scrape_details'] = []
	  for chart_dict in chart_dict_list:
	    temp_dict = chart_dict.copy()
	    
	    url = temp_dict['url']
	    url = url.replace('bitcoin', coin_dict['full_name'])
	    url = url.replace('btc', coin_dict['coin'])
	    url = url.replace(' ', '%20')

	    temp_dict['url'] = url
	    coin_dict['scrape_details'].append(temp_dict)

	coin_merged_df_list = []

	for coin_dict in coin_dict_list:
	  coin_df_list = []
	  for page in coin_dict['scrape_details']:
	    try:
	      coin_df_list.append(get_bitinfochart_graph_values(url=page['url'], var_name=page['name']))
	    except:
	      empty_df = pd.DataFrame()
	      empty_df['full_name'] = coin_dict['full_name']
	      empty_df['coin'] = coin_dict['coin']
	      coin_df_list.append(pd.DataFrame)

	  coin_df = merge_dfs(coin_df_list)
	  coin_df['full_name'] = coin_dict['full_name']
	  coin_df['coin'] = coin_dict['coin']

	  coin_merged_df_list.append(coin_df)

	  clear_output()

	  #if not COMBINE_ALL:
	  #  file_path = RESULTS_FOLDER + '/' +coin_dict['full_name'] + '.csv'
	  #  coin_df.to_csv(file_path)
	  
	#if COMBINE_ALL:
	combined_df = pd.concat(coin_merged_df_list, ignore_index=True, sort=False)
	combined_df['date'] = pd.to_datetime(combined_df['date'])
	combined_df.set_index('date', inplace=True)
	combined_df=combined_df.sort_index()
	prices=combined_df.price
	prices=prices.dropna()
	prices=np.array(prices)
	price=[]
	for i in prices:
	  price.append(float(i))

	predicted_values=[]
	days=[1,7,30,90]
	for i in days:
		if(i==day):
			predicted_values.append(next_day_price[0])
		else:
			model=pickle.load(open("randomforestregressor_"+str(i)+".sav", 'rb'))
			predicted_values.append(model.predict(test_point.reshape(1,-1))[0])
	#getting the last 10 days values
	price=price[-50:]
	predicted_values.insert(0,priceUSD)
	predicted_values.insert(0,price[-1])
	price=pd.DataFrame(price)
	price.index=range(-50,0)
	predicted_values=pd.DataFrame(predicted_values)
	predicted_values.index=(-1,0,1,7,30,90)
	global future
	global true
	true=price
	future=predicted_values
	forecast_day=day
	forecast_price=next_day_price
	return render_template('predict.html',prediction_text="Today Price: "+str(priceUSD)+"      | "+str(day)+"th day Price: "+str(next_day_price[0]))


@app.route('/visualize')
def visualize():
	global true
	global future
	global forecast_day
	global forecast_price
	print("future:")
	print(future)
	print("true:")
	print(true)
	fig,ax=plt.subplots(figsize=(6,4))
	ax=plt.axes()
	forecast_price=forecast_price[0]	
	print("forecast price",forecast_price,"forecast_day",forecast_day)
	#ax.annotate(text='Forecast Day', xy=(forecast_day, forecast_price), xytext=(forecast_day+10, forecast_price-10) ,arrowprops= dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=90"))
	ax.plot(true,label='True')
	ax.plot(future,label='Future')
	plt.xlabel('Days')
	plt.ylabel('PriceUSD')
	plt.legend()
	plt.grid()
	canvas=FigureCanvas(fig)
	img=io.BytesIO()
	fig.savefig(img)
	img.seek(0)
	return send_file(img,mimetype='img/png')


def parse_strlist(sl):
	clean = re.sub("[\[\],\s]","",sl)
	splitted = re.split("[\'\"]",clean)
	values_only = [s for s in splitted if s != '']
	return values_only

def get_bitinfochart_graph_values(url, var_name):
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')

  scripts = soup.find_all('script')
  for script in scripts:
      if 'd = new Dygraph(document.getElementById("container")' in script.text:
          StrList = script.text
          StrList = '[[' + StrList.split('[[')[-1]
          StrList = StrList.split(']]')[0] +']]'
          StrList = StrList.replace("new Date(", '').replace(')','')
          dataList = parse_strlist(StrList)

  date = []
  value = []
  for each in dataList:
      if (dataList.index(each) % 2) == 0:
          date.append(each)
      else:
          value.append(each)

  df = pd.DataFrame(list(zip(date, value)), columns=["date",var_name])
  return df

def merge_dfs(df_list):
  df_merged = None
  for i in range(len(df_list)-1):
    if i == 0:
      df_merged = df_list[i].merge(df_list[i+1], on='date', how='outer')
    else:
      df_merged = df_merged.merge(df_list[i+1], on='date', how='outer')

  return df_merged


if __name__ == "__main__":
    app.run(debug=True)
