#import numpy
from flask import Flask,request,jsonify, render_template
import pickle
from bs4 import BeautifulSoup 
import requests
import lxml
from pickle import load
import numpy as np
app=Flask(__name__)


@app.route('/')
def home():
	return 	render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	try:
		day = request.form['prediction']
	except:
		return render_template('index.html',prediction_text="Please choose a forecast day!")
	model=pickle.load(open("randomforestregressor_"+str(day)+".sav", 'rb'))
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
	#['median_transaction_feeUSD','hashrate','sentinusdUSD','difficulty','priceUSD']
	test_point=np.array([median_transaction_fee,hashrate,sentinusd,difficulty,priceUSD])
	next_day_price=model.predict(test_point.reshape(1,-1))

	return render_template('index.html',prediction_text="Today Price: "+str(priceUSD)+"      | "+str(day)+"th day Price: "+str(next_day_price[0]))


if __name__ == "__main__":
    app.run(debug=True)
