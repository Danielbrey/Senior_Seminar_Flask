from flask import Flask, render_template

from multipage import MultiPage
from pages import average_demand #, day_average_demand, real_time, bihall # import your pages here
#from pages.average_demand import app

app = Flask(__name__)

@app.route('/home')
def home():
   return render_template('homepage.html')

@app.route('/bihall')
def bihall():
   return render_template('bihall.html')

@app.route('/test')
def test():
   return render_template('average_demand.html', title = "Test")





@app.route('/chart/')
def chart():
   return average_demand.app()

if __name__ == '__main__':
   app.run()
