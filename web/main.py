from flask import Flask, render_template, request, redirect, make_response
import sys
import os
import hashlib
import loginData as db
import sessions
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
global web_url
web_url = "https://HTM2021.aceharvey.repl.co"

@app.route("/")
def index():
    return render_template("index.html")
@app.route('/incorrect-login')
def inclogin():
 return render_template("inclogin.html")
@app.route('/username-Taken')
def nametaken():
 return render_template("nametaken.html")
@app.route('/homepage')
def homepage():
 return render_template("homepage.html", username = sessions.decode(request.cookies.get('usr')))
@app.route("/greenday")
def greenday():
  return render_template("greenday.html")
@app.route("/login")
def login():
  return render_template("login.html")
@app.route("/register")
def register():
  return render_template("register.html")
  
@app.route("/upload")
def upload():
  return render_template("upload.html")
@app.route("/collection")
def collection():
  return render_template("collection.html")

@app.route('/registerpage', methods=["POST"])
def verif_R():
  if request.method == 'POST':
    loginn = request.form.getlist("uname")[0]
    loginp = request.form.getlist("psw")[0]
    resp = make_response(render_template("verif_R.html"))
    if not db.sqlexists(loginn, "n_u_l_l"):
      salt = os.urandom(32)
      loginp_hashed = hashlib.pbkdf2_hmac('sha256',loginp.encode('utf-8'),salt,100000)
      loginp_hashed = salt + loginp_hashed
      db.insertintotable(loginn, loginp_hashed)
      return redirect(
        web_url + "/", code=302)
    else:
      return redirect(
        web_url + "/username-Taken", code=302)
    return resp

@app.route('/verif-page', methods=["POST"])
def verif():
  if request.method == 'POST':
    loginn = request.form.getlist("uname")[0]
    loginp = request.form.getlist("psw")[0]
    conn = db.create_connection("data.db")
    c = conn.cursor()
    t = (loginn,)
    c.execute('SELECT * FROM Logins WHERE Usernames=?;',t)
    conn.commit()
    a = c.fetchone()
    conn.commit()
    conn.close()
    print(a)
    try:
      loginp = hashlib.pbkdf2_hmac('sha256',loginp.encode('utf-8'),a[1][:32],100000)
    except:
      return redirect(
          web_url + "/incorrect-login", code=302)
    if loginp == a[1][32:]:
      resp = make_response(redirect(web_url + "/homepage"))
      resp.set_cookie("usr", loginn)
      resp.set_cookie("session_identifier", sessions.newsession(a[0]))
      return resp
    else:
      return redirect(
        web_url + "/incorrect-login", code=302)
  else:
    return redirect(web_url, code=302)

@app.route('/uploadIMG', methods=["POST"])
def uploadIMG():
  print(request.files.keys())
  for each in request.files.values():
    print(each)
  if request.method == 'POST':
    try:
      img = Image.open(request.files["img"])
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Input
        from tensorflow.keras.layers import Lambda
        from tensorflow.keras.datasets import mnist
        import tensorflow as tf
        import os
        import numpy as np
        import sys

        model = tf.keras.models.load_model('output/siamese_model') #needs the "output" folder from colab in /web/
        classLabels = ["labels of classes list"] #this needs to be working
        classes = ["list of control images from test vids/drink as a file location"]
        x=[]
        for each in classes:
          x.append(model.predict([np.assaray(Image.open(each)), preprocess(img)], batch_size=15)[0][0])
        y=max(x)

        return "can = " + classLabels.index[y]
      #indetify the can here! it sus! sus can! report it vented >:((
    except Exception as e:
      print(e)
    return redirect(web_url+"/homepage", code=302)
#greyscale
def get_grayscale(image):
    return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

#canny edge detection
def canny(image):
  
  return cv2.Canny(image, 100, 200)
def preprocess(x):
  import numpy as np
  return np.assaray(canny(get_grayscale(x)))
def main():
  db.init()
  app.run(host='0.0.0.0', port=8080)
main()