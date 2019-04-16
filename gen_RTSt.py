from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import scipy.io as sio
import pydicom
import os
import gen_RTDose
from matplotlib.path import Path
from skimage import measure, transform
def inpolygon(xq, yq, xv, yv):
    """
    reimplement inpolygon in matlab
    :type xq: np.ndarray
    :type yq: np.ndarray
    :type xv: np.ndarray
    :type yv: np.ndarray
    """
    # 合并xv和yv为顶点数组
    vertices = np.vstack((xv, yv)).T
    # 定义Path对象
    path = Path(vertices)
    # 把xq和yq合并为test_points
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    # 得到一个test_points是否严格在path内的mask，是bool值数组
    _in = path.contains_points(test_points)
    # 得到一个test_points是否在path内部或者在路径上的mask
    _in_on = path.contains_points(test_points)
    # 得到一个test_points是否在path路径上的mask
    _on = _in ^ _in_on
    return _in_on
def matchContour(RS_file,Slice,index):
    numberofROI = len(index)       #靶区个数
    mub = []
    for i in range(numberofROI):    ##选取靶区
        ite = 0
        nub = index[i]          ## 靶区序号
        if nub == -1:
            print("存在-1情况,靶区顺序号: "+str(i))
            print(RS_file.PatientName)
        try:
            numberofcoutours = len(RS_file.ROIContourSequence[nub].ContourSequence) #每一层的靶区层数
            for j in range(numberofcoutours):
                csContent = RS_file.ROIContourSequence[nub].ContourSequence[j]
                dcmname = 'CT.' + csContent.ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm'
                x = dcmname.split('.')
                x = int(x[-2])
                if Slice == x:
                    ite += 1
                    if ite == 1:
                        nubslc = str(j)
                    # mub.append((nub,j,ite))
                    if ite >1 :
                        nubslc = nubslc+' '+str(j)
                        # print(str(Slice)+' '+str(nub)+' '+nubslc)
                        print("\t","Slice: [%2d], 靶区顺序号: [%2d], ROI contour靶区序号: [%2d]," \
                              % (Slice,i,nub)+ 'Contour序号:'+nubslc)
                        print("\t",RS_file.PatientName)
            if ite == 0:
                mub.append([-1,-1,0])
            else:
                mub.append([nub,nubslc,ite])
        except(AttributeError):
            mub.append((-1,-1,0))
            print('no ContourSequence'+' '+str(nub))
            print(RS_file.PatientName)
    return mub
def rectcontour(RS_file,dcm_file, pairCont,indexList):
    img = np.zeros((512,512))
    dcmOrigin = dcm_file.ImagePositionPatient
    dcmSpacing = dcm_file.PixelSpacing
    numberofROI = len(indexList)  ##靶区的个数

    img_sum = np.zeros((512,512,numberofROI))
    for i in range(numberofROI):
        ROIC = pairCont[i]
        w, l = ROIC[0], ROIC[1]         ## w为对应靶区序号，l为靶区对应contoursequence
        ite = ROIC[2]
        if ite == 1:
            l = int(l)
            csContent = RS_file.ROIContourSequence[w].ContourSequence[l]      ##靶区第i层信息
            numberofPoint = int(csContent.NumberOfContourPoints)  # 该层靶区的曲线点数
            wldPosition = np.zeros((numberofPoint, 3))  # 靶区曲线的物理坐标
            picPosition = np.zeros((numberofPoint, 2))  # 靶区曲线的图像空间
            for jj in range(numberofPoint):
                ii = jj * 3
                wldPosition[jj,0] = csContent.ContourData[ii]   #物理坐标
                wldPosition[jj,1] = csContent.ContourData[ii+1]
                wldPosition[jj,2] = csContent.ContourData[ii+2]
                picPosition[jj, 0] = np.round((wldPosition[jj, 0] - dcmOrigin[0]) / dcmSpacing[0])  # 轮廓x坐标
                picPosition[jj, 1] = np.round((wldPosition[jj, 1] - dcmOrigin[1]) / dcmSpacing[1])  # 轮廓y坐标
            x = list(range(512))
            y = x
            [X, Y] = np.meshgrid(x, y)
            x = X.flatten()
            y = Y.flatten()
            x_cont = picPosition[:, 0]
            y_cont = picPosition[:, 1]
            mask = inpolygon(x,y,x_cont,y_cont)
            mask = np.reshape(mask,(512,512))
            mask_no = ~mask
            img[mask] = 255
            img[mask_no] = 0
            img_sum[:,:,i] = img
            # plt.imshow(img, cmap='gray')
            # plt.plot(x_cont,y_cont)
            # plt.show()
        elif ite > 1:
            print(w)
            lList = list(map(int,l.split(' ')))
            mask_sum = np.zeros([512,512,ite],dtype=bool)
            # j = ite - 1
            for chf in range(ite):
                # ROIC = pairCont[i + chf]
                # w, l = ROIC[0], ROIC[1]
                aa = lList[chf]
                csContent = RS_file.ROIContourSequence[w].ContourSequence[aa]  ##靶区第i层信息
                numberofPoint = int(csContent.NumberOfContourPoints)  # 该层靶区的曲线点数
                wldPosition = np.zeros((numberofPoint, 3))  # 靶区曲线的物理坐标
                picPosition = np.zeros((numberofPoint, 2*ite))  # 靶区曲线的图像空间
                for jj in range(numberofPoint):
                    ii = jj * 3
                    wldPosition[jj, 0] = csContent.ContourData[ii]  # 物理坐标
                    wldPosition[jj, 1] = csContent.ContourData[ii + 1]
                    wldPosition[jj, 2] = csContent.ContourData[ii + 2]
                    picPosition[jj, 0] = np.round((wldPosition[jj, 0] - dcmOrigin[0]) / dcmSpacing[0])  # 轮廓x坐标
                    picPosition[jj, 1] = np.round((wldPosition[jj, 1] - dcmOrigin[1]) / dcmSpacing[1])  # 轮廓y坐标
                x = list(range(512))
                y = x
                [X, Y] = np.meshgrid(x, y)
                x = X.flatten()
                y = Y.flatten()
                x_cont = picPosition[:, 0]
                y_cont = picPosition[:, 1]
                mask = inpolygon(x, y, x_cont, y_cont)
                mask = np.reshape(mask, (512, 512))
                # mask_no = ~mask
                # img[mask] = 255
                # img[mask_no] = 0
                # plt.imshow(img, cmap='gray')
                # plt.plot(x_cont,y_cont)
                # plt.show()
                mask_sum[:,:,chf] = mask
            maskFs = mask_sum[:,:,0]
            for ii in range(1,ite):
                maskLin = mask_sum[:,:,ii]
                maskFs = maskFs^maskLin
            mask_no = ~maskFs
            img[maskFs] = 255
            img[mask_no] = 0
            img_sum[:,:,i] = img
            # plt.imshow(img, cmap='gray')
            # # plt.plot(x_cont,y_cont)
            # plt.show()
        # plt.imshow(img_sum[:,:,i],cmap='gray')
        # plt.show()
    return img_sum
def readfile(INPUT_FOLDER,index,slice):
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    ## 选取患者
    patient_path = INPUT_FOLDER + '/' + patients[index]
    one_patient = os.listdir(patient_path)
    # Rs_file = pydicom.read_file(patient_path + '/' + one_patient[-1])  ##读取RS文件
    nubSlice = len(one_patient)-3
    ###指定切片读取
    spl = one_patient[0].split('.')
    sliceStr = str(slice)
    spl[-2] = sliceStr
    slicePath = '.'.join(spl)
    dcm_flie = pydicom.read_file(patient_path+'/'+slicePath)
    ## 所有切片读取
    return dcm_flie
def matchIndex(Rs_file,targetName):     #匹配器官的ROInumber,获得
    index = [-1]*6
    ROINumberSu = [-1]*6
    structRoISequence = Rs_file.StructureSetROISequence
    ROIContourSq = Rs_file.ROIContourSequence
    for i in range(len(structRoISequence)):
        ROIName = structRoISequence[i].ROIName
        if ROIName in targetName:
            wz = targetName.index(ROIName)
            ROINumber = structRoISequence[i].ROINumber
            ROINumberSu[wz] = ROINumber
    for j in range(len(ROIContourSq)):
        ReROINumber = ROIContourSq[j].ReferencedROINumber
        if ReROINumber in ROINumberSu:
            wz2 = ROINumberSu.index(ReROINumber)
            index[wz2] = j

    return index
def RTDose(patientPath,dcm_file,slice,img):
    nub = img.shape[-1]
    DoseRido = [0,0]
    dcmRido = [0,0]
    dcmPicPosLu = [0,0]
    dcmPicPosRido = [0,0]
    patientList = os.listdir(patientPath)
    RTDoseFile = pydicom.read_file(patientPath+'/'+patientList[-3])
    NubFram = RTDoseFile.NumberOfFrames
    indexDose = (NubFram-1)-2*(slice-1)
    ArrayDose = RTDoseFile.pixel_array
    DoseMat = ArrayDose[indexDose]
    DoseMatScale = transform.resize(DoseMat,(512,512))
    # DoseMatScale = cv.resize(DoseMat,(512,512))
    DoseWldOrigin = RTDoseFile.ImagePositionPatient
    DoseSpace = RTDoseFile.PixelSpacing
    DoseRow = RTDoseFile.Rows
    DoseColu = RTDoseFile.Columns
    DoseRido[0] = DoseWldOrigin[0]+(DoseColu-1)*DoseSpace[0]
    DoseRido[1] = DoseWldOrigin[1]+(DoseRow-1)*DoseSpace[1]
    dcmWldOrigin = dcm_file.ImagePositionPatient
    dcmSpace = dcm_file.PixelSpacing
    dcmColu = dcm_file.Columns
    dcmRow = dcm_file.Rows
    dcmRido[0] = dcmWldOrigin[0]+(dcmColu-1)*dcmSpace[0]
    dcmRido[1] = dcmWldOrigin[1]+(dcmRow-1)*dcmSpace[1]
    dcmPicPosLu[0] = round((DoseWldOrigin[0]-dcmWldOrigin[0])/dcmSpace[0])
    dcmPicPosLu[1] = round((DoseWldOrigin[1]-dcmWldOrigin[1])/dcmSpace[1])
    dcmPicPosRido[0] = round(dcmColu-(dcmRido[0]-DoseRido[0])/dcmSpace[0])
    dcmPicPosRido[1] = round(dcmRow-(dcmRido[1]-DoseRido[1])/dcmSpace[1])
    for i in range(nub):
        imgz = img[:,:,i]
        imgz = imgz[dcmPicPosLu[1]:dcmPicPosRido[1],dcmPicPosLu[0]:dcmPicPosRido[0]]
        imgz = cv.resize(imgz, (512, 512), interpolation=cv.INTER_CUBIC)
        img[:,:,i] = imgz
    # plt.imshow(imgz,cmap='gray')
    # plt.show()
    # plt.imshow(DoseMat)
    # plt.show()
    return img,DoseMatScale
def readRs(INPUT_FOLDER,index):
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    ## 选取患者
    patient_path = INPUT_FOLDER + '/' + patients[index]
    one_patient = os.listdir(patient_path)
    Rs_file = pydicom.read_file(patient_path + '/' + one_patient[-1])  ##读取RS文件
    nubSlice = len(one_patient)-3
    return Rs_file,nubSlice,patient_path


# path = 'E:/Data/CervialDose'
# a = list(range(12))
# b = list(range(16,24))
# soulie = a + b
# for ii in range(len(soulie)):
#     indexPatient = soulie[ii]        #患者索引，即第几个患者
#     Rs_file, nubSlc, patient_path = readRs(path,indexPatient)
#     ArrDose = gen_RTDose.recontRD(patient_path)
#     for m in range(nubSlc):
#         Slice = m+1      #文件后缀，即第几个切片
#         targetName = ['BODY','BLADDER','femoral_head_l','femoral_head_r','RECTUM','PTV']
#         dcm_file = readfile(path,indexPatient,Slice)
#         indexContour = matchIndex(Rs_file,targetName)
#         pairCont = matchContour(Rs_file,Slice,indexContour)
#         img_ret = rectcontour(Rs_file,dcm_file,pairCont,indexContour)
#         savepath = './data/data.' + str(ii) + '.' + str(Slice) + '.mat'
#         sio.savemat(savepath, {'RTSt': img_ret, 'RTDose': ArrDose[m]})
        # np.savez('./data/data.' + str(ii) + '.' + str(Slice) + '.npz', RTSt=img_ret, RTDose=ArrDose[m])
# RTDoseFlie = RTDose(patientPath,dcm_file,Slice,img_ret)
