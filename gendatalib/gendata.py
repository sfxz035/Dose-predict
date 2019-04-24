import scipy.io as sio
from gendatalib import gen_RTDose
from gendatalib import gen_RTSt
import numpy as np
import matplotlib.pyplot as plt
# from gendatalib.gen_RTSt import *
# from gendatalib.gen_RTDose import *
def test():
    Test_dir = 'E:/code/Unet/gendatalib/data1/train/data.33.79.npz'
    data_test = np.load(Test_dir)
    Test_input = data_test['RTSt']
    Test_label = data_test['RTDose']
    for i in range(6):
        plt.imshow(Test_input[:,:,i])
        plt.show()
if __name__ == '__main__':
    test()
    path = 'E:/Data/CervialDose'
    # soulie = list(range(24))
    # soulie.remove(6)
    # soulie.remove(7)
    # soulie.remove(14)
    # soulie.remove(20)
    # soulie = a + b
    soulie = list(range(24,46))
    for ii in range(len(soulie)):
        indexPatient = soulie[ii]        #患者索引，即第几个患者
        # indexPatient = 6
        Rs_file, nubSlc, patient_path = gen_RTSt.readRs(path,indexPatient)
        ArrDose = gen_RTDose.recontRD(patient_path)
        for m in range(nubSlc):
            Slice = m+1      #文件后缀，即第几个切片
            # Slice = 55
            targetName = ['BODY','BLADDER','FEMORAL_HEAD_L','FEMORAL_HEAD_R','RECTUM','PTV']
            dcm_file = gen_RTSt.readfile(path,indexPatient,Slice)
            indexContour = gen_RTSt.matchIndex(Rs_file,targetName)
            # if -1 in indexContour:
            #     print("存在 -1情况,患者号: " + str(indexPatient))
            #     break
            pairCont = gen_RTSt.matchContour(Rs_file,Slice,indexContour)
            img_ret = gen_RTSt.rectcontour(Rs_file,dcm_file,pairCont,indexContour)
            # sio.savemat('./data/data.'+str(ii)+'.'+str(Slice)+'.mat', {'RTSt': img_ret,'RTDose:':ArrDose[:,:,m]})
            # np.savez('./data1/train/data.'+str(indexPatient)+'.'+str(Slice)+'.npz',RTSt=img_ret,RTDose=ArrDose[:,:,m])
            print("患者序号： [%2d], 切片序号：[%2d]. "\
                    %(indexPatient,Slice))