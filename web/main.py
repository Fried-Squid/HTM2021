from flask import Flask, render_template, request, redirect, make_response
import sys
import os
import hashlib
import loginData as db
import sessions

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
 return render_template("homepage.html")

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
      resp.set_cookie("session_identifier", sessions.newsession(a[0],"b"))
      return resp
    else:
      return redirect(
        web_url + "/incorrect-login", code=302)
  else:
    return redirect(web_url, code=302)


def main():
  db.init()
  app.run(host='0.0.0.0', port=8080)
main()