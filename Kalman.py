from re import L
from matplotlib.pyplot import flag
import numpy as np
import math
from matplotlib import pyplot as pt
import pandas as pd
import logging
import sys
from datetime import date
today = date.today()
Presentdate = today.strftime("%d-%m-%Y")

#path = "C:\\myproj\\"+Presentdate+"_log_data.log"
# logging.basicConfig(filename=path, filemode='a', level=logging.DEBUG,
#                   format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
# print(prediction)


def sums(a):
    a1 = 0
    for i, v in enumerate(a):
        a1 = a1+v
    a2 = len(a)
    a1 = a1/a2
    return a1


def Kalman(prevpred, prevact, prediction):
    index = 0
    error1 = []
    flag = []

    for i, v in enumerate(prevact):
        res = [prevact[i + 1] - prevact[i] for i in range(len(prevact)-1)]
    for i in res:
        index = i+index
    le = len(res)
    # print(res)
    um = index/(le)
    #print('_____', um)

    for i in range(len(prevact)):
        temp = (prevact[i]-prevpred[i])/prevact[i]
        error1.append(temp)
        if(temp < 0):
            flag.append(0)
        else:
            flag.append(1)

    Avg = 0
    for i in range(len(error1)):
        Avg += error1[i]
    Avg1 = len(error1)
    up = Avg/Avg1
    # print(up)
    # print(up)

    #logging.info('Average Error = ', up)
    #logging.info('Uncertainity Measurement = ', um)
    # print(um)

    Kalmangain = um/(um+up)
    #logging.info('Kalmangain = ', Kalmangain)
    # print(Kalmangain,"kg")

    pp = sums(prediction)
    mpre = sums(prevact)
    prpre = sums(prevpred)
    C_Data1 = []
    count=0
    count1=0
    for i in range(len(prediction)):
     if prevact[i]-prevpred[i]<=100:
        count+=1
     elif prevact[i]-prevpred[i]<=0:
        count1+=1                
     if len(count) > 5:
        C_Data1.append(prediction[i])
     else:
        if count1>3:
         C_Data1.append(prediction[i]-(Kalmangain*(prevact[i]-prevpred[i])))
        else:
         C_Data1.append(prediction[i]+(Kalmangain*(prevact[i]-prevpred[i])))
        
    # print(rightact)
    # print(C_Data)

    #Sq_Avg = 0
    # for i in range(len(C_Data1)):
    #    Sq_Avg += (C_Data1[i]*C_Data1[i])
    #Sq_Avg /= len(C_Data1)
    # print(Sq_Avg)
    #logging.info('Squared Average = ', Sq_Avg)
    #RMSE = round(math.sqrt(Sq_Avg), 2)
    # print("rmse=",RMSE)
    #logging.info('Root Mean Square Error = ', RMSE)
    # print(len(C_Data1))
    return C_Data1
