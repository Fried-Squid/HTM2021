import sqlite3

def create_connection(db_file): #cant remember what this does but my database code relies on it
  conn = None
  try:
    conn = sqlite3.connect(db_file)
    return conn
  except Exception as e:
    global le
    le = e
    print(e)
  return conn

def init():
  conn = create_connection("data.db") #creates the table if it doesnt exist
  c = conn.cursor()
  c.execute('CREATE TABLE IF NOT EXISTS Logins (Usernames VARCHAR(255), Passwords VARCHAR(255));')
  conn.commit()
  conn.close()

def insertintotable(u, p):           #inserts a given vector of username, password
  conn = create_connection("data.db")
  c = conn.cursor()
  t = (u, p)
  c.execute('INSERT INTO Logins (Usernames, Passwords) VALUES (?,?)', t)
  conn.commit()
  conn.close()

def sqlexists(u, p):                #checks if a given vector is a username/password combination.
  if p != "n_u_l_l":
    conn = create_connection("data.db")
    c = conn.cursor()
    t = (u, p)
    c.execute('SELECT EXISTS(SELECT 1 FROM Logins WHERE Usernames=? AND Passwords=?);',t)
    conn.commit()
    a = c.fetchone()
    conn.commit()
    conn.close()
    print(a)
    if a != (0, ):
      return True
    else:
      return False
  else:
    conn = create_connection("data.db")
    c = conn.cursor()
    c.execute('SELECT EXISTS(SELECT 1 FROM Logins WHERE Usernames=?);',(u, ))
    conn.commit()
    a = c.fetchone()
    conn.commit()
    print(a)
    if a != (0, ):
      return True
    else:
      return False