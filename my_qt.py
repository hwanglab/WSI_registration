import sys
import os
from PyQt5.QtWidgets import QApplication, QFileDialog,QWidget

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        # self.openFileNameDialog()
        # self.openFileNamesDialog()
        # self.saveFileDialog()
        # self.openDirecotryDialog()        
        # self.show()

    def openDirecotryDialog(self, prefix :str):

        dir = QFileDialog.getExistingDirectory(self,"QFileDialog.getExistingDirectory()", os.getcwd() if not prefix else prefix)
        return dir

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
    
    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)
    
    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)



def dir_QT(prefix=os.getcwd()):
    app = QApplication(sys.argv) 
    ex = App()  
    dir = ex.openDirecotryDialog(prefix)
    # sys.exit(app.exec_())

    return dir
