import itk
import numpy as np
import matplotlib.pyplot as plt


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'Created Done')
        return True
    else:
        print(path + 'Exist')
        return False
# dicomPath = 'E:/Data/cancer_data/708469chenxiuhui'
def recontRD(dicomPath):
    # ..................................................................................................
    #                                    Read DICOM Series
    # ..................................................................................................
    # typedef imagetype
    ImageType = itk.Image[itk.F, 3]
    # read dicom"
    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction('0')
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dicomPath)
    seriesUID = namesGenerator.GetSeriesUIDs()
    for uid in seriesUID:
        seriesIdentifier = uid
        break
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.Update()
    image = reader.GetOutput()
    # print(image)
    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    fileNames = namesGenerator.GetFileNames(seriesUID[1])
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.Update()
    imageDicom = reader.GetOutput()
    # print(imageDicom)

    # ..................................................................................................
    #                                    resample
    # ..................................................................................................
    startIndex = itk.Index[3]()
    startIndex.Fill(0)
    # resampleidentity
    resampleFilter = itk.ResampleImageFilter[ImageType, ImageType].New()
    # new parameters
    regionDicom = imageDicom.GetLargestPossibleRegion()
    originResample = imageDicom.GetOrigin();
    spacingResample = imageDicom.GetSpacing()
    sizeResample = regionDicom.GetSize()

    resampleFilter.SetSize(sizeResample)
    resampleFilter.SetOutputOrigin(originResample)
    resampleFilter.SetOutputSpacing(spacingResample)
    resampleFilter.SetOutputStartIndex(startIndex)
    resampleFilter.SetDefaultPixelValue(0)
    resampleFilter.SetInput(image)
    resampleFilter.Update()
    image = resampleFilter.GetOutput()
    # print(image)
    # ..................................................................................................
    #                                     for get slice data
    # ..................................................................................................

    ImageType2dF = itk.Image[itk.F, 2]
    ImageType2dC = itk.Image[itk.UC, 2]
    sliceSize = image.GetBufferedRegion().GetSize()
    slizeIndex = image.GetBufferedRegion().GetIndex()
    totalNum = sliceSize[2]
    # mkdir(dicomPath + "\\PNG_IMAGES" )
    ArrDose = np.zeros([sliceSize[0],sliceSize[1],sliceSize[2]])
    # ArrDose = []
    for i in range(totalNum):
        extractImageFilter = itk.ExtractImageFilter[ImageType, ImageType2dF].New()
        extractImageFilter.SetDirectionCollapseToIdentity()
        extractImageFilter.SetInput(image)
        # print("All image numbers are: " + str(totalNum) + " Now is: " + str(i))
        sliceSize[2] = 0
        slizeIndex[2] = i
        regionDesired = itk.ImageRegion[3]()
        regionDesired.SetIndex(slizeIndex)
        regionDesired.SetSize(sliceSize)
        extractImageFilter.SetExtractionRegion(regionDesired)
        extractImageFilter.Update()
        extractImageFilter.Update()
        imageDose = extractImageFilter.GetOutput()
        arr = itk.GetArrayViewFromImage(imageDose)
        ArrDose[:,:,totalNum-1-i] = arr
        # ArrDose.append(arr)
    # ArrDose.reverse()
    # plt.imshow(ArrDose[0])
    # plt.show()
    return ArrDose
        # # ..............................................................................................
        # #                              Window Level
        # # ..............................................................................................
        # windowLevelFilter = itk.IntensityWindowingImageFilter[ImageType2dF, ImageType2dC].New()
        # windowLevelFilter.SetInput(extractImageFilter.GetOutput())
        # windowLevelFilter.SetOutputMaximum(255)
        # windowLevelFilter.SetOutputMinimum(0)
        # windowLevelFilter.SetWindowLevel(255, 40)
        # # ..............................................................................................
        # #                              write to image file
        # # ..............................................................................................
        # writePath = dicomPath + "/PNG_IMAGES/" + seriesIdentifier + "_" + str(i) + ".png"
        # writer = itk.ImageFileWriter[ImageType2dC].New()
        # writer.SetInput(windowLevelFilter.GetOutput())
        # writer.SetFileName(writePath)
        # writer.Update()


# Path = 'E:/Data/CervialDose/17010507 tong xiu yu'
# ArrDose = recontRD(Path)
