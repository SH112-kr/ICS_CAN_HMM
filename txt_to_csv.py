import pandas as pd

def normal_txt_to_csv(txt,csv): #정상 txt경로 및 저장할 csv 경로 
    file = pd.read_csv(txt, delimiter = '\t',header=None,names=['TimeStamp','Arbitation','DLC','DATA','CLASS1'])
    file.to_csv(csv)
    
def abnormal_txt_to_csv(txt,csv): #비정상 txt경로 및 저장할 csv 경로 
    file = pd.read_csv(txt, delimiter = '\t',header=None,names=['TimeStamp','Arbitation','DLC','DATA','CLASS1','CLASS2'])
    file.to_csv(csv)

abnormal_txt_to_csv('C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (attack included).txt','C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (attack included).csv')
normal_txt_to_csv('C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (normal only).txt','C:\\Users\\SSH\\Desktop\\python_code\\python\\Center\\Apk_Malware_Crawler\\ICS_CAN\\CAN traffic (normal only).csv')