from flask import Flask, render_template

from multipage import MultiPage
from pages import average_demand #, day_average_demand, real_time, bihall # import your pages here


app = Flask(__name__)

@app.route('/home')
def home():
   return render_template('homepage.html', title="Welcome to Green Midd")

@app.route('/average_demand')
def avg_demand():
   return average_demand.app()

@app.route('/bihall')
def bihall():
   return render_template('bihall.html', title="Bicentennial Hall")









if __name__ == '__main__':
   app.run()
