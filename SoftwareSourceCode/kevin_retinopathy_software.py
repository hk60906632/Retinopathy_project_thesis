from PyQt4 import QtGui, uic
from PyQt4 import QtCore
from PyQt4.QtGui import QGraphicsView, QRubberBand, QImage, QPainter, QWidget, QLabel, QVBoxLayout, QPixmap
from PyQt4.QtCore import QThread, QRect, pyqtSignal, QPoint, QSize
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt4agg import (
        FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from sklearn import linear_model
from itertools import zip_longest
from timeit import default_timer as timer
import sys
import cv2
import csv
import os
import numpy as np

######Python version 3.6.9, PyQT4
#this is the main window class, it contains all the widgets of the main window and connect
#all the logic to all the widgets
class Window(QtGui.QMainWindow):

    def __init__(self, parent=None):
        super(Window, self).__init__()
        uic.loadUi("mainWindow.ui", self)
        self.setWindowTitle("Kevin Retinopathy project")
        self.progressBar.setValue(0)
        self.Graphs.clicked.connect(self.setUpGraphPage)
        self.Load.clicked.connect(self.file_open)
        self.selectROI.clicked.connect(self.callROIpage)
        self.FluorscentIntensity.clicked.connect(self.getFluorescentIntensity)
        self.selectGradientROI.clicked.connect(self.callGradientROIpage)
        self.saveAll.clicked.connect(self.saveAllasCSV)
        self.saveSharp.clicked.connect(self.saveSharpasCSV)
        self.saveBlur.clicked.connect(self.saveBlurasCSV)
        self.SaveGradientGraph.clicked.connect(self.saveGradientCSV)
        self.quit.clicked.connect(self.close_application)
        self.getGradientButton.clicked.connect(self.getGradientFromROI)
        self.resetButtonAdvanceSet.clicked.connect(self.resetAdvanceSetting)
        self.KmeanIterSpinBox.setRange(1, 15)
        self.KmeanIterSpinBox.setValue(7)
        self.gettingFluroscentIntensity = False
        self.GradientROIpageEXIST = False
        #set up the PyQT combo boxes
        self.kmean_Eps_List = ["0.1", "0.5", "1", "1.5", "2"]
        self.KmeansEpsilon.addItems(self.kmean_Eps_List)
        self.sobel_kernel_list = ["7", "5", "3"]
        self.SobelKernalSizeComboBox.addItems(self.sobel_kernel_list)
        self.meanbox_list = ["20", "10", "30", "40", "50", "60", "70", "80", "90", "100" ]
        self.BRMeanBoxComboBox.addItems(self.meanbox_list)
        self.ptpbox_list = ["10", "4", "8", "20", "30", "40", "50", "60", "70", "80", "90", "100"]
        self.BRPtPBoxComboBox.addItems(self.ptpbox_list)
        self.tolerance_list = ["0.1", "0", "0.05", "0.5", "1", "1.5", "2"]
        self.BRtoleranceComboBox.addItems(self.tolerance_list)

    def closeEvent(self, *args, **kwargs):
        super(QtGui.QMainWindow, self).closeEvent(*args, **kwargs)
        sys.exit()

    def close_application(self):
        sys.exit()

    def resetAdvanceSetting(self):
        self.KmeanIterSpinBox.setValue(7)
        self.KmeansEpsilon.setCurrentIndex(self.kmean_Eps_List.index("0.1"))
        self.SobelKernalSizeComboBox.setCurrentIndex(self.sobel_kernel_list.index("7"))
        self.BRMeanBoxComboBox.setCurrentIndex(self.meanbox_list.index("20"))
        self.BRPtPBoxComboBox.setCurrentIndex(self.ptpbox_list.index("10"))
        self.BRtoleranceComboBox.setCurrentIndex(self.tolerance_list.index("0.1"))

    def file_open(self):
        if self.gettingFluroscentIntensity == False:
            self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
            self.videoLocation.setText(self.filename)
            #print(self.filename)
            cap = cv2.VideoCapture(self.filename)
            totalFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.X1spinBox.setMaximum(self.width)
            self.X2spinBox.setMaximum(self.width)
            self.Y1spinBox.setMaximum(self.height)
            self.Y2spinBox.setMaximum(self.height)
            self.videoWidthLabel.setText(str(self.width))
            self.videoHeightLabel.setText(str(self.height))
            self.totalNumFrames.setText(str(totalFrame-1))
            self.userSelectFrameNumber.setMaximum(totalFrame)
            # print(self.width)
            # print(self.height)


    def getFluorescentIntensity(self):
        if (self.X1spinBox.value()==0 and self.X2spinBox.value()==0 and self.Y1spinBox.value()==0 and self.Y2spinBox.value()==0):
            return

        if self.gettingFluroscentIntensity == False:
            self.fluorscentIntensity = VideoAnalysis(self)
            self.fluorscentIntensity.start()
            self.gettingFluroscentIntensity = True
            self.connect(self.fluorscentIntensity, QtCore.SIGNAL('FIProgress'), self.updateProgressBar)
            self.connect(self.fluorscentIntensity, QtCore.SIGNAL('FIFinish'), self.allowGetFI)



    def updateProgressBar(self, val):
        self.progressBar.setValue(val)


    def allowGetFI(self):
        self.gettingFluroscentIntensity = False
        self.gradStartSpinbox.setMaximum(int(max(self.fluorscentIntensity.sharpData[0:len(self.fluorscentIntensity.sharpData), 0]))+1)
        self.gradEndSpinBox.setMaximum(int(max(self.fluorscentIntensity.sharpData[0:len(self.fluorscentIntensity.sharpData), 0]))+1)

    def callROIpage(self):
        if self.gettingFluroscentIntensity == False:
            cap = cv2.VideoCapture(self.filename)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.userSelectFrameNumber.value())
            res, frame = cap.read()
            self.pickROIframe = SelectROI(frame)
            self.pickROIframe.show()
            self.connect(self.pickROIframe.band, QtCore.SIGNAL('selectingROI'), self.updateCoordination)



    def updateCoordination(self):
        if(int(self.pickROIframe.band.x1)>self.width):
            self.X1spinBox.setValue(int(self.width))
        elif(int(self.pickROIframe.band.x1)<0):
            self.X1spinBox.setValue(0)
        else:
            self.X1spinBox.setValue(int(self.pickROIframe.band.x1))

        if(int(self.pickROIframe.band.x2)>self.width):
            self.X2spinBox.setValue(int(self.width))
        elif(int(self.pickROIframe.band.x2)<0):
            self.X2spinBox.setValue(0)
        else:
            self.X2spinBox.setValue(int(self.pickROIframe.band.x2))

        if(int(self.pickROIframe.band.y1)>self.height):
            self.Y1spinBox.setValue(int(self.height))
        elif(int(self.pickROIframe.band.y1)<0):
            self.Y1spinBox.setValue(0)
        else:
            self.Y1spinBox.setValue(int(self.pickROIframe.band.y1))

        if(int(self.pickROIframe.band.y2)>self.height):
            self.Y2spinBox.setValue(int(self.height))
        elif(int(self.pickROIframe.band.y2)<0):
            self.Y2spinBox.setValue(0)
        else:
            self.Y2spinBox.setValue(int(self.pickROIframe.band.y2))


    def setUpGraphPage(self):
        if self.gettingFluroscentIntensity == False:
            fig1 = Figure()
            f1 = fig1.add_subplot(111)
            f1.scatter(self.fluorscentIntensity.sharpData[0:len(self.fluorscentIntensity.sharpData), 0],
                        self.fluorscentIntensity.sharpData[0:len(self.fluorscentIntensity.sharpData), 1], 1)
            f1.set_title('Sharp Fluorscent Intensity vs Time')
            f1.set_xlabel('Seconds')
            f1.set_ylabel('Fluorscent Intensity')

            fig2 = Figure()
            f2 = fig2.add_subplot(111)
            f2.scatter(self.fluorscentIntensity.sharpData[0:len(self.fluorscentIntensity.sharpData), 0],
                        self.fluorscentIntensity.sharpData[0:len(self.fluorscentIntensity.sharpData), 2], 1)
            f2.set_title('Sharp Edge-Sharpness vs Time')
            f2.set_xlabel('Seconds')
            f2.set_ylabel('Edge-Sharpness')


            fig3 = Figure()
            f3 = fig3.add_subplot(111)
            f3.scatter(self.fluorscentIntensity.allData[0:len(self.fluorscentIntensity.allData), 0],
                        self.fluorscentIntensity.allData[0:len(self.fluorscentIntensity.allData), 1], 1)
            f3.set_title('Fluorscent Intensity vs Time')
            f3.set_xlabel('Seconds')
            f3.set_ylabel('Fluorscent Intensity')


            fig4 = Figure()
            f4 = fig4.add_subplot(111)
            f4.scatter(self.fluorscentIntensity.allData[0:len(self.fluorscentIntensity.allData), 0],
                        self.fluorscentIntensity.allData[0:len(self.fluorscentIntensity.allData), 2], 1)
            f4.set_title('Edge-Sharpness vs Time')
            f4.set_xlabel('Seconds')
            f4.set_ylabel('Edge-Sharpness')

            fig5 = Figure()
            f5 = fig5.add_subplot(111)
            f5.scatter(self.X, self.y, 1,c='b')
            f5.plot(self.line_X, self.line_y_ransac, color='r')
            f5.set_title('RANSAC Best fit of selected ROI')
            f5.set_xlabel('Seconds')
            f5.set_ylabel('Fluorscent Intensity')

            self.graph = GraphPage()
            self.graph.addfig('Sharp Fluorscent Intensity vs Time', fig1)
            self.graph.addfig('Sharp Edge-Sharpness vs Time', fig2)
            self.graph.addfig('Fluorscent Intensity vs Time', fig3)
            self.graph.addfig('Edge-Sharpness vs Time', fig4)
            self.graph.addfig('RANSAC Best Fit', fig5)
            self.graph.show()

    def callGradientROIpage(self):
        if self.gettingFluroscentIntensity == False and self.GradientROIpageEXIST == False:
            self.GradientROIpageEXIST = True
            self.gradientROI = SelectGradientROI(self.fluorscentIntensity.sharpData)
            self.gradientROI.show()
            self.connect(self.gradientROI, QtCore.SIGNAL('closedGradientROIpage'), self.allowCallGradROI)
            self.connect(self.gradientROI, QtCore.SIGNAL('selectedGradientROI'), self.getGradientFromROI)

    def allowCallGradROI(self):
        self.GradientROIpageEXIST = False
        self.gradStartSpinbox.setValue(self.gradientROI.Xmin)
        self.gradEndSpinBox.setValue(self.gradientROI.Xmax)



    def getGradientFromROI(self):
        #print("called getGradientFromROI")
        self.ROIGradFI = []
        self.ROIGradTime = []
        # print(self.gradStartSpinbox.value())
        # print(self.gradEndSpinBox.value())
        for x in self.fluorscentIntensity.sharpData:
            if x[0] > self.gradStartSpinbox.value() and x[0] < self.gradEndSpinBox.value():
                self.ROIGradTime.append(x[0])
                self.ROIGradFI.append(x[1])
        self.ROIGradTime = np.asarray(self.ROIGradTime).reshape(-1, 1)
        self.ROIGradFI =  np.asarray(self.ROIGradFI).reshape(-1, 1)
        self.X = self.ROIGradTime
        self.y = self.ROIGradFI
        self.ransac = linear_model.RANSACRegressor(random_state=0).fit(self.X, self.y)
        self.line_X = np.arange(self.X.min(), self.X.max())[:, np.newaxis]
        self.line_y_ransac = self.ransac.predict(self.line_X)
        # print(float(self.ransac.estimator_.coef_))
        self.gradientValue.setText(str(float(self.ransac.estimator_.coef_)))

    def saveAllasCSV(self):
        if self.gettingFluroscentIntensity == False:
            name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
            data = self.fluorscentIntensity.allData
            with open(name, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(("Time(seconds)", "fluorscentIntensity_ratio", "totalEdge", "veinIntensity", "exchangeIntensity"))
                writer.writerows(data)
            csvFile.close()

    def saveSharpasCSV(self):
        if self.gettingFluroscentIntensity == False:
            name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
            sharpData = self.fluorscentIntensity.sharpData
            with open(name, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(("Time(seconds)", "fluorscentIntensity_ratio", "totalEdge", "veinIntensity", "exchangeIntensity"))
                writer.writerows(sharpData)
            csvFile.close()

    def saveBlurasCSV(self):
        if self.gettingFluroscentIntensity == False:
            name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
            blurData = self.fluorscentIntensity.blurData
            with open(name, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(("Time(seconds)", "fluorscentIntensity_ratio", "totalEdge", "veinIntensity", "exchangeIntensity"))
                writer.writerows(blurData)
            csvFile.close()

    def saveGradientCSV(self):
        if self.gettingFluroscentIntensity == False:
            name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
            totalData = []
            roiGradTime = [float(x) for x in self.ROIGradTime]
            roiGradFI =  [float(y) for y in self.ROIGradFI]
            totalData.append(roiGradTime)
            totalData.append(roiGradFI)
            ransacX = [float(a) for a in self.line_X]
            ransacY = [float(b) for b in self.line_y_ransac]
            totalData.append(ransacX)
            totalData.append(ransacY)
            gradient = float(self.ransac.estimator_.coef_)
            gradientList = [gradient]
            totalData.append(gradientList)
            exportData = zip_longest(*totalData, fillvalue = ' ')

            with open(name, 'w', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(("Time(seconds)", "Fluorscent_Intensity_Ratio", "Ransac_fit_X", "Ransac_fit_Y", "Gradient"))
                writer.writerows(exportData)
            csvFile.close()


#this is the class that start a new thread to apply Kmeans clustering and Sobel edge detection to all the frames of the video
#this class also used to separate all the data into sharp and blur sets of data
class VideoAnalysis(QThread):
    # def __init__(self, name, iteration, epsilon, kernel_size, meanbox_size, ptpbox_size, tolerance, y1, y2, x1, x2):
    def __init__(self, parent):
        QtCore.QThread.__init__(self, parent=None)
        self.allData = []
        self.sharpData = []
        self.blurData = []
        self.name = parent.filename
        self.y1 = parent.Y1spinBox.value()
        self.y2 = parent.Y2spinBox.value()
        self.x1 = parent.X1spinBox.value()
        self.x2 = parent.X2spinBox.value()
        self.kmeansIter = parent.KmeanIterSpinBox.value()
        self.kmeansEps = float(parent.KmeansEpsilon.currentText())
        self.Kernel = int(parent.SobelKernalSizeComboBox.currentText())
        self.MeanBox = int(int(parent.BRMeanBoxComboBox.currentText())/2)
        self.PtpBox = int(int(parent.BRPtPBoxComboBox.currentText())/2)
        self.tolerance = float(parent.BRtoleranceComboBox.currentText())


    def run(self):
        startTime = timer()
        # print("Iteration :" + str(self.kmeansIter))
        # print("epsilon :" + str(self.kmeansEps))
        # print("x1 :" + str(self.x1))
        # print("x2 :" + str(self.x2))
        # print("y1 :" + str(self.y1))
        # print("y2 :" + str(self.y2))
        currentFrame = 0
        cap = cv2.VideoCapture(self.name)
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        # print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : "+ str(self.fps))

        ret, frame = cap.read()
        videolength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_img = grayFrame[self.y1:self.y2,
                                    self.x1:self.x2]
        imgArray = crop_img.ravel()
        width = imgArray.size
        imgArray = imgArray.reshape((width, 1))
        imgArray = np.float32(imgArray)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 7, 0.1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        init_conpactness,init_labels,init_centers = cv2.kmeans(imgArray,2, None, criteria, 7, flags)
        centerA = init_centers
        labelsA = init_labels

        while(True):
            ret, frame = cap.read()
            if not ret:
                break

            name = str(currentFrame)
            #print("Analyzing frame..." + name)

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop_img = grayFrame[self.y1:self.y2,
                                    self.x1:self.x2]
            #equalizedGray = cv2.equalizeHist(crop_img)
            imgArray = crop_img.ravel()
            width = imgArray.size
            imgArray = imgArray.reshape((width, 1))
            imgArray = np.float32(imgArray)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.kmeansIter, self.kmeansEps)
            flags = cv2.KMEANS_USE_INITIAL_LABELS
            conpactness,labels,centers = cv2.kmeans(imgArray, 2, labelsA, criteria, self.kmeansIter, flags, centerA)
            veinIntensity = float(max(centers))
            exchangeIntensity = float(min(centers))
            fluorscentIntensity = float(min(centers)/ max(centers))


            sobelx = cv2.Sobel(crop_img, cv2.CV_64F, 1, 0, ksize=self.Kernel)
            sobely = cv2.Sobel(crop_img, cv2.CV_64F, 0, 1, ksize=self.Kernel)
            sobelxSq = np.square(sobelx.ravel())
            sobelySq = np.square(sobely.ravel())
            totalEdge = np.sum(np.sqrt(np.add(sobelxSq, sobelySq)))/((self.y2 - self.y1)*(self.x2 - self.x1))
            #print(veinIntensity)
            self.allData.append([currentFrame, fluorscentIntensity, totalEdge, veinIntensity, exchangeIntensity])
            currentFrame += 1
            self.progress = (currentFrame/(videolength-1))*100
            #print("CenterA : "+ str(centerA))
            centerA = centers
            labelsA = labels
            self.emit(QtCore.SIGNAL('FIProgress'), self.progress)
        self.allData = np.asarray(self.allData)

        for x in self.allData:
            x[0] = x[0]/self.fps
        self.blurRemoval()
        duration = timer() - startTime
        # print(duration)


    def blurRemoval(self):
        # print("Kernel: " + str(self.Kernel))
        # print("Tolerance: " + str(self.tolerance))
        # print("Meanbox: " + str(self.MeanBox))
        # print("PtpBox: " + str(self.PtpBox))
        tolerance_of_blur = int(np.average(self.allData[0:len(self.allData), 2])*self.tolerance)
        for i in range(60, len(self.allData)):
            #print("removing blur---" + str(i))
            mean = np.mean(self.allData[i-self.MeanBox:i+self.MeanBox, 2])
            if np.ptp(self.allData[i-self.PtpBox:i+self.PtpBox, 2]) > tolerance_of_blur:
                if self.allData[i][2]>= mean:
                    self.sharpData.append(self.allData[i])

                else:
                    self.blurData.append(self.allData[i])

            else:
                self.sharpData.append(self.allData[i])


        self.sharpData = np.asarray(self.sharpData)
        self.blurData = np.asarray(self.blurData)
        self.emit(QtCore.SIGNAL('FIFinish'))


#this class responsible for creating the resizible rectangular selector
#on the selected frame to allow user to select the reagion of interest of video frame
class ResizableRubberBand(QtGui.QWidget):
    def __init__(self, parent=None):
        super(ResizableRubberBand, self).__init__(parent)

        self.draggable = True
        self.dragging_threshold = 5
        self.mousePressPos = None
        self.mouseMovePos = None
        self.borderRadius = 5

        self.setWindowFlags(QtCore.Qt.SubWindow)
        layout = QtGui.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            QtGui.QSizeGrip(self), 0,
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        layout.addWidget(
            QtGui.QSizeGrip(self), 0,
            QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self._band = QtGui.QRubberBand(
            QtGui.QRubberBand.Rectangle, self)
        self._band.show()
        self.show()


    def resizeEvent(self, event):
        self._band.resize(self.size())
        self.x1 = self.pos().x()
        self.y1 = self.pos().y()
        self.x2 = self.x1 + self.size().width()
        self.y2 = self.y1 + self.size().height()
        # print("x1 --> " + str(self.x1))
        # print("x2 --> " + str(self.x2))
        # print("y1 --> " + str(self.y1))
        # print("y2 --> " + str(self.y2))
        self.emit(QtCore.SIGNAL('selectingROI'))

    def paintEvent(self, event):
        # Get current window size
        window_size = self.size()
        qp = QtGui.QPainter()
        qp.begin(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, True)
        qp.drawRoundedRect(0, 0, window_size.width(), window_size.height(),
                           self.borderRadius, self.borderRadius)
        qp.end()
        self.x1 = self.pos().x()
        self.y1 = self.pos().y()
        self.x2 = self.x1 + self.size().width()
        self.y2 = self.y1 + self.size().height()
        # print("x1 --> " + str(self.x1))
        # print("x2 --> " + str(self.x2))
        # print("y1 --> " + str(self.y1))
        # print("y2 --> " + str(self.y2))
        self.emit(QtCore.SIGNAL('selectingROI'))

    def mousePressEvent(self, event):
        if self.draggable and event.button() == QtCore.Qt.RightButton:
            self.mousePressPos = event.globalPos()                # global
            self.mouseMovePos = event.globalPos() - self.pos()    # local
        super(ResizableRubberBand, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draggable and event.buttons() & QtCore.Qt.RightButton:
            globalPos = event.globalPos()
            moved = globalPos - self.mousePressPos
            self.x1 = self.pos().x()
            self.y1 = self.pos().y()
            self.x2 = self.x1 + self.size().width()
            self.y2 = self.y1 + self.size().height()
            # print("x1 --> " + str(self.x1))
            # print("x2 --> " + str(self.x2))
            # print("y1 --> " + str(self.y1))
            # print("y2 --> " + str(self.y2))
            self.emit(QtCore.SIGNAL('selectingROI'))
            if moved.manhattanLength() > self.dragging_threshold:
                # Move when user drag window more than dragging_threshold
                diff = globalPos - self.mouseMovePos
                self.move(diff)
                self.mouseMovePos = globalPos - self.pos()
        super(ResizableRubberBand, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mousePressPos is not None:
            if event.button() == QtCore.Qt.RightButton:
                moved = event.globalPos() - self.mousePressPos
                if moved.manhattanLength() > self.dragging_threshold:
                    # Do not call click event or so on
                    event.ignore()
                self.mousePressPos = None
        super(ResizableRubberBand, self).mouseReleaseEvent(event)


#this class responsible to load and display the user's desire frame and put the rectangular selector on top of it
class SelectROI(QtGui.QMainWindow):
    def __init__(self, SelectFrame, parent=None):
        super(SelectROI, self).__init__(parent)
        img = SelectFrame
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        lay = QVBoxLayout(self.central_widget)
        self.label = QLabel(self)
        height, width, byteValue = img.shape
        self.setFixedSize(width, height)
        colorImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.mQImage = QImage(colorImg.data, colorImg.shape[1], colorImg.shape[0], colorImg.shape[1]*3, QImage.Format_RGB888)
        pixmap = QPixmap(self.mQImage)
        self.label.setPixmap(pixmap)
        lay.addWidget(self.label)
        self.band = ResizableRubberBand(self.label)
        self.band.setGeometry(0, 0, 150, 150)
        # print(img.shape)


    def closeEvent(self, *args, **kwargs):
        super(QtGui.QMainWindow, self).closeEvent(*args, **kwargs)
        self.emit(QtCore.SIGNAL('selectingROI'))


class GraphPage(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(GraphPage, self).__init__()
        uic.loadUi("graphPage.ui", self)
        self.fig_dict = {}
        self.listWidget.itemClicked.connect(self.changeFig)
        fig = Figure()
        self.addplot(fig)

    def addplot(self, fig):
        self.canvas = FigureCanvas(fig)
        self.verticalLayout.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self.widget, coordinates=True)
        self.verticalLayout.addWidget(self.toolbar)

    def rmPlot(self):
        self.verticalLayout.removeWidget(self.canvas)
        self.canvas.close()
        self.verticalLayout.removeWidget(self.toolbar)
        self.toolbar.close()

    def addfig(self, name, fig):
        self.fig_dict[name] = fig
        self.listWidget.addItem(name)

    def changeFig(self, item):
        text = item.text()
        self.rmPlot()
        self.addplot(self.fig_dict[text])

class SelectGradientROI(QtGui.QMainWindow):
    def __init__(self, sharpData ,parent=None):

        super(SelectGradientROI, self).__init__(parent)

        #plot the Large Vessel Fluorescent Itensity vs Time && Exchange Vessel Fluorescent Itensity vs Time
        self.figure = Figure()
        self.figure.subplots_adjust(hspace=0.5)
        self.veinVSframe = self.figure.add_subplot(211)
        self.veinVSframe.scatter(sharpData[0:len(sharpData), 0], sharpData[0:len(sharpData), 3])
        self.veinVSframe.set_title('Large Vessel Fluorescent Itensity vs Time')
        self.veinVSframe.set_xlabel('Seconds')
        self.veinVSframe.set_ylabel('Large Vessel Fluorescent Itensity')
        self.exchangeVSframe = self.figure.add_subplot(212)
        self.exchangeVSframe.scatter(sharpData[0:len(sharpData), 0], sharpData[0:len(sharpData), 4])
        self.exchangeVSframe.set_title('Exchange Vessel Fluorescent Itensity vs Time')
        self.exchangeVSframe.set_xlabel('Seconds')
        self.exchangeVSframe.set_ylabel('Exchange Vessel Fluorescent Itensity')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.span = SpanSelector(self.veinVSframe, self.onselect, 'horizontal', useblit=True,
                    rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)
        self.Xmin = 0
        self.Xmax = 0

        #add the plotted graphs to the layout so can be seen on the screen
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.canvas.draw()

    def onselect(self, xmin, xmax):
        #print("Xmin = %lf Xmax = %lf" % (xmin, xmax))
        self.Xmin = xmin
        self.Xmax = xmax


    def closeEvent(self, *args, **kwargs):
        super(QtGui.QMainWindow, self).closeEvent(*args, **kwargs)
        self.emit(QtCore.SIGNAL('closedGradientROIpage'))
        #only emit signal when user select some region
        if (self.Xmin >= 0 and self.Xmax > 0):
            self.emit(QtCore.SIGNAL('selectedGradientROI'))



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())
