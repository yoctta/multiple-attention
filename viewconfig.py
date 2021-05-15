import sys
import pickle
import fire
import glob
import json
import os

def loadconfig(a):
  name='runs/{}/config.pkl'.format(a)
  with open(name,'rb') as f:
    a=pickle.load(f)
  return a

def printconfig(a):
  for i in vars(a):
    print(i,getattr(a,i))


def cateval(a):
  files=glob.glob('evaluations/%s/metrics-*.json'%a)
  rt=[]
  for i in files:
    with open(i) as f:
      rt.append(json.load(f))
  return rt

filter1=lambda x:x['ff']['all']['frame_acc']

def main():
  l=os.listdir('runs')
  d={}
  for i in l:
    conf=loadconfig(i)
    if conf.attention_layer=='b5':
      v=list(map(filter1,cateval(i)))
      if v:
        d[i]=max(v)
  print(d)
  n=max(d,key=lambda x:d[x])
  print(n,d[n])
  printconfig(loadconfig(n))

if __name__=="__main__":
  fire.Fire()


