﻿# ICS_CAN_HMM
사용방법

pip3 install -r requirements.txt 로 필요한 모듈 다운로드
.py와 데이터셋이 같이있는 경로에서 cmd 명령어로 python CAN_HMM.py -h 를 치면 도움말이 나옵니다.

제 python 버전은 3.9.6입니다.


명령어 예시
C:\Users\SSH\Desktop\python_code\python\Center\Apk_Malware_Crawler\ICS_CAN>python CAN_HMM.py -e "CAN traffic (normal only).csv" -t "CAN traffic (attack included).csv" -o 2 -s 3 -w 5

      -------------
        DATA Result
      -------------


--------------------------------------------------------------------------
Training log :  -36481929.51881357
Test log     :  -165542766.14767802
윈도우 사이즈 5  의Traing log 와 Test log의 거리 :  129060836.62886444
초기 상태 확률 :  [0.00000000e+000 8.12785138e-100 1.00000000e+000]
전이 확률 행렬 :
 [[9.11139880e-03 6.97699380e-01 2.93189221e-01]
 [8.13036250e-01 1.86948480e-01 1.52701493e-05]
 [1.34585050e-05 1.49649084e-01 8.50337457e-01]]
--------------------------------------------------------------------------

C:\Users\SSH\Desktop\python_code\python\Center\Apk_Malware_Crawler\ICS_CAN>python CAN_HMM.py -e "CAN traffic (normal only).csv" -t "CAN traffic (attack included).csv" -o 1 -s 3 -w 5

      -------------------
       Arbitration Result
      -------------------


--------------------------------------------------------------------------
Training log :  -6202921.989993358
Test log     :  -29051384.990707528
윈도우 사이즈 5  의Traing log 와 Test log의 거리 :  22848463.000714168
초기 상태 확률 :  [3.73966230e-08 3.83792302e-19 9.99999963e-01]
전이 확률 행렬 :
 [[8.25624369e-01 2.21262209e-18 1.74375631e-01]
 [1.71580376e-01 7.35989828e-01 9.24297967e-02]
 [1.96786319e-08 2.52021721e-01 7.47978259e-01]]
--------------------------------------------------------------------------
