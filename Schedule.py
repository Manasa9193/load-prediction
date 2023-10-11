import schedule
import time
import Realtime_Forecast as RTF
import DayAheadPrediction as DAP
# DAP.LSTM_MAIN()
DAP.LSTM_MAIN()
#RTF.Kalman_F()
#schedule.every().day.at("23:30").do(DAP.LSTM_MAIN)
#schedule.every(1).hours.do((RTF.Kalman_F))
'''
while True:
    schedule.run_pending()
    time.sleep(5*60)
    '''