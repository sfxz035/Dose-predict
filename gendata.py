import scipy.io as sio
import gen_RTDose
import gen_RTSt
import numpy as np


if __name__ == '__main__':
    path = 'E:/Data/CervialDose'
    soulie = list(range(24))
    soulie.remove(6)
    soulie.remove(7)
    soulie.remove(14)
    soulie.remove(20)
    # soulie = a + b
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
            # np.savez('./data/data.'+str(ii)+'.'+str(Slice)+'.npz',RTSt=img_ret,RTDose=ArrDose[:,:,m])
            print(str(indexPatient)+' '+str(m))