import sys
import os
from infant import *
from PyQt5 import QtWidgets, QtGui, QtCore

class MyForm(QtWidgets.QMainWindow):
  def __init__(self,parent=None):
     QtWidgets.QWidget.__init__(self,parent)
     self.ui = Ui_MainWindow()
     self.ui.setupUi(self)
     self.ui.pushButton.clicked.connect(self.bcacc)
     self.ui.pushButton_2.clicked.connect(self.nnacc)
     self.ui.pushButton_3.clicked.connect(self.nnpred)
     self.ui.pushButton_4.clicked.connect(self.bcpred)

  def bcacc(self):
    os.system("python -W ignore bg.py")

  def nnacc(self):
    os.system("python -W ignore nn1.py")

  def nnpred(self):
    os.system("python -W ignore nn2.py")

  def bcpred(self):  
    os.system("python -W ignore bc2.py")

       
if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
