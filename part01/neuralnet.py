import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
L = [3,5,1]               # Network
EPSILON = 5.0             # Learning Rate
EPOCH = 1000000           # Epoch
TOLER = 0.01              # Tolerance Limit
BATCH = False             # Batch Learning [True/False]
ACTYPE = 'sigmoid'        # Activate Function [sigmoid/tanh/relu]
INITWEIGHT = 'PlusMinus'  # Initializing Weights [PlusMinus/Gauss1,0.1,0.01/Xavier/He]

# Plot Trace
LOG_ERROR = True         # Logging Error [True/False]
LOG_WIGHT = True         # Logging Weight [True/False]

# Input Data
XDATA = np.array([ 
    [ 0, 0, 0 ],
    [ 1, 0, 1 ],
    [ 1, 1, 1 ],
    [ 1, 1, 0 ],
    [ 1, 0, 0 ],
    [ 0, 0, 1 ]
])

# Teaching Data
TDATA = np.array([ 
    [ 0 ],
    [ 0 ],
    [ 1 ],
    [ 0 ],
    [ 1 ],
    [ 1 ]
])

YDATA = {}                     # layer outputs 
dEdS = {}                      # dE/dS (E:Error S:Sum)
W = {}                         # Weight
fe = open('error.log', 'wb')   # File Pinter of Error Log
fw = open('weight.log', 'wb')  # File Pinter of Weight Log

# Weight Initialization
for layer in range(len(L) - 1):
  if INITWEIGHT == "PlusMinus":
    W[layer] = np.arange((L[layer]+1)*L[layer+1]).reshape((L[layer]+1,L[layer+1])) % 2
    W[layer][W[layer]==0] = -1
  elif INITWEIGHT == "Gauss1":
    W[layer] = np.random.randn(L[layer], L[layer+1]) * 1
    W[layer] = np.vstack((W[layer], np.array([np.zeros(L[layer+1])])))
  elif INITWEIGHT == "Gauss0.1":
    W[layer] = np.random.randn(L[layer], L[layer+1]) * 0.1
    W[layer] = np.vstack((W[layer], np.array([np.zeros(L[layer+1])])))
  elif INITWEIGHT == "Gauss0.01":
    W[layer] = np.random.randn(L[layer], L[layer+1]) * 0.01
    W[layer] = np.vstack((W[layer], np.array([np.zeros(L[layer+1])])))
  elif INITWEIGHT == "Xavier":
    W[layer] = np.random.randn(L[layer], L[layer+1]) / np.sqrt(L[layer])
    W[layer] = np.vstack((W[layer], np.array([np.zeros(L[layer+1])])))
  elif INITWEIGHT == "He":
    W[layer] = np.random.randn(L[layer], L[layer+1]) / np.sqrt(L[layer]) * np.sqrt(2.0)
    W[layer] = np.vstack((W[layer], np.array([np.zeros(L[layer+1])])))
  else:
    exit()

def activate(x_):
  if ACTYPE == 'sigmoid':
    return 1.0 / (1.0 + np.exp(-x_))  # sigmoid
  elif ACTYPE == 'tanh':
    return np.tanh(x_)
  elif ACTYPE == 'relu':
    return np.maximum(0,x_)

def dactivate(x_):
  if ACTYPE == 'sigmoid':
    return (1.0 - x_) * x_
  elif ACTYPE == 'tanh':
    return 1.0 - np.tanh(x_) * np.tanh(x_)
  elif ACTYPE == 'relu':
    d = np.zeros_like(x_)
    d[x_>=0] = 1
    return d

def feedforward():

  YDATA[0] = XDATA

  for layer in range(len(L)-1): 
    l0 = layer
    l1 = layer + 1
    YDATA[l0] = np.hstack([YDATA[l0], np.ones(XDATA.shape[0]).reshape(XDATA.shape[0],1)])
    YDATA[l1] = np.dot(YDATA[l0], W[l0])
    YDATA[l1] = activate(YDATA[l1]) 

  y = YDATA[len(L)-1]

  # 2乗和誤差の平均値を保持
  err = np.average(((y - TDATA)**2), axis=1)
  #err = np.average(np.absolute(y - TDATA), axis=1)

  if LOG_ERROR :
    np.savetxt(fe, np.array([err]))

  return err

def feedbackward():
  y = YDATA[len(L)-1]
  dEdS[len(L)-1] = (y - TDATA) * dactivate(y)

  for layer in range( (len(L) - 1), 1, -1):
    l0 = layer - 1
    l1 = layer
    dEdS[l0] = np.dot(W[l0], dEdS[l1].T )
    dEdS[l0] = dEdS[l0].T * dactivate(YDATA[l0])
    dEdS[l0] = dEdS[l0][::,:-1]

def weightupdate(td=None):

  for layer in range( len(L) - 1 ):
    if( BATCH ):
      dEdW = np.dot(YDATA[layer].T, dEdS[layer+1])
    else:
      dEdW = (YDATA[layer][td].reshape(L[layer]+1, 1) * dEdS[layer+1][td])

    W[layer] = W[layer] - EPSILON * dEdW

  if LOG_WIGHT :
    for k,v in W.items():
      np.savetxt(fw, v.flatten(), newline=' ')
    np.savetxt(fw, np.array([0]))

# Start Learning
for epc in range(EPOCH):

  if(BATCH):        ### 一括重み更新 ###
    err = feedforward()
    feedbackward()
    weightupdate()
  else:            ### 逐次的重み更新(SGD) ###
    for td in range( XDATA.shape[0] ):
      err = feedforward()
      feedbackward()
      weightupdate(td) 

  # 終了判定
  if np.max(err) < TOLER:
    break

  if epc % 100 == 0:
    print( "epoch=", epc )

fe.close()
fw.close()
print( "!END!=", epc )


# 以下波形プロット
if LOG_ERROR :
  fe = np.loadtxt("error.log")
  fe = fe.T
  plt.subplot(len(L)*100+11)
  for tr in range( fe.shape[0] ):
    plt.plot(range( fe.shape[1] ), fe[tr], label="tr "+str(tr))
  plt.ylabel("Error")
  plt.legend()

if LOG_WIGHT :
  fw = np.loadtxt("weight.log")
  base = 0;
  for k,v in W.items():
    fw0 = fw[::,base:base+v.size].T
    plt.subplot(len(L)*100+10+k+2)
    plt.ylabel("W[" + str(k) + "]")
    for i in range( fw0.shape[0] ):
      plt.plot(range( fw0.shape[1] ), fw0[i], label=str((int)(i/v.shape[1]))+":"+str(i%v.shape[1]) )
    base += v.size
    plt.legend()

if LOG_ERROR or LOG_WIGHT :
  plt.xlabel("weight updates")
  plt.show()
