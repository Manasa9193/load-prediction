# Yet to Complete

# Get today's prediction data of LSTM and Previous day actual data
# Check if time (if in 00:00 to 01:00)
# Send 60 Samples of previous day actual data and 60 samples of today's prediction data to Kalman Filter to predict 1st 60 samples.
# if not betwn 00:00 to 01:00
# Send the previous hr data of prediction and actual and next hr data of prediction of sameday to Kalman Filter to predict next 60 samples
import Kalman as Klm
import util as ut
from datetime import datetime
import DBData as DB
from datetime import timedelta
now = datetime.now()
import pandas as pd


S_Time = '00:00'
E_Time = '01:00'
N_Time = now.strftime("%H:%M")

E_Time = datetime.strptime(E_Time, "%H:%M")
S_Time = datetime.strptime(S_Time, "%H:%M")
N_Time = datetime.strptime(N_Time, "%H:%M")

# Get today's prediction data of LSTM and Previous day actual data
today1 = datetime.today()
yday = today1 - timedelta(1)
next_hr = today1 + timedelta(hours=1)


def Kalman_F():
    
    present_hr = str(now.hour)
    next_hr = now + timedelta(hours=1)
    # if(is_time_between(time(0,0),time(1,0))):# Check if time (if in 00:00 to 01:00)
    if ut.isNowInTimePeriod(S_Time, E_Time, N_Time):
        # Send 60 Samples of previous day actual data and 60 samples of today's prediction data to Kalman Filter to predict 1st 60 samples.
        # Get Prev day actual data
        # Today's LSTM data
        prev_predicted = DB.getdata(0, '00:00', yday)
        prev_actual = DB.getdata(1, '00:00', yday)
        predicted = DB.getdata(0, '00:00', today1)
        prev_predicted1=DB.getdata(0, '00:00', today1)
        predicted1=DB.getdata(0, '01:00', today1)
        print("In if block")
        res = Klm.Kalman(prev_predicted, prev_actual, predicted)
        res1=Klm.kalman(prev_predicted1,res,predicted1)
                

    else:
        present_hr = str(now.hour)
        next_hr = now + timedelta(hours=1)
        prehour = now - timedelta(hours=1)
        pre_hour = str(prehour.hour)
        print("In elif block", present_hr)
        # Get present_hr-1 data of actual
        prev_predicted = DB.getdata(0, pre_hour, today1)
        prev_actual = DB.getdata(1, pre_hour, today1)
        predicted = DB.getdata(0, present_hr, today1)
        prev_predicted1=DB.getdata(0, present_hr, today1)
        predicted1=DB.getdata(0, next_hr, today1)
        #print(len(predicted), len(prev_actual), len(prev_predicted))
        res = Klm.Kalman(prev_predicted, prev_actual, predicted)
        res1= Klm.Kalman(prev_predicted1,res,predicted1)
    res_list = [*res, *res1]
    



    DB.save_dt(res, present_hr, today1)
    DB.save_dt(res1, next_hr, today1)

    return(res_list)


#def Kalman_F():

def KL():
    res_list = Kalman_F()
    
    # save(res)
