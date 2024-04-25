import sys
import PyQt5.QtWidgets as qw
import PyQt5.QtGui as qg
import numpy as np
import cv2 as cv
import Recognition
import shutil
import os
import Extractor as fp
import Compare as fc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class RecognitionFrom(qw.QWidget):
    def __init__(self):
        super().__init__()

        self.ui = Recognition.Ui_Recognition()
        self.ui.setupUi(self)
        self.ui_init()

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = qw.QVBoxLayout(self.ui.groupBox_4)
        layout.addWidget(self.canvas)

        self.rocData = []
        self.fingerprint_for_match = None
        self.matchName = None
        self.vm = None
        self.ls = None
    
    def ui_init(self):
        self.ui.pushButton_Enroll.clicked.connect(self.enroll_fingerprint)
        self.ui.pushButton__Match.clicked.connect(self.match_fingerprint)
        self.ui.pushButton_Evaluate.clicked.connect(self.evaluate_performance)
        self.ui.pushButton_Clear.clicked.connect(self.clear_text)
        self.ui.pushButton_Draw.clicked.connect(self.draw_roc_curve)
    
    def clear_text(self):
        self.ui.textEdit_Results.setPlainText("")

    def enroll_fingerprint(self):
        name, ok = qw.QInputDialog.getText(self, 'Enroll', 'Enter your name:')
        if ok:
            fingerprint_path, _ = qw.QFileDialog.getOpenFileName(self, os.getcwd())
            if fingerprint_path:
                fingerprint_folder = os.path.join(os.getcwd(), 'sample')
                if not os.path.exists(fingerprint_folder):
                    os.makedirs(fingerprint_folder)
                new_png_fingerprint_path = os.path.join(fingerprint_folder, f'{name}.png')
                new_npz_fingerprint_path = os.path.join(fingerprint_folder, f'{name}.npz')
                shutil.copy(fingerprint_path, new_png_fingerprint_path)
                
                fingerprint = fp.Fingerprint(new_png_fingerprint_path)
                fingerprint.fingerprint_process()
                fingerprint.detect_minutiae_pos_and_dir()
                fingerprint.create_local_structures()
                get_valid_minutiae = fingerprint.valid_minutiae
                # print("Get valid minutiae:", get_valid_minutiae)
                get_local_structures = fingerprint.local_structures
                # print("Get local structures:", get_local_structures)
                
                valid_minutiae_array = np.array(get_valid_minutiae, dtype=object)

                np.savez(new_npz_fingerprint_path, get_valid_minutiae=valid_minutiae_array, get_local_structures=get_local_structures)
                '''data = np.load(new_npz_fingerprint_path, allow_pickle=True)
                valid_minutiae_data = data['get_valid_minutiae']
                local_structures_data = data['get_local_structures']
                print("valid_minutiae:", valid_minutiae_data)
                print("local_structures", local_structures_data)'''

                qw.QMessageBox.information(self, 'Enrollment',  f'Fingerprint enrolled for {name} successfully!')

    def match_fingerprint(self):
        fingerprint_for_match, _ = qw.QFileDialog.getOpenFileName(self, os.getcwd())
        if fingerprint_for_match:  
            getFeatures = fp.Fingerprint(fingerprint_for_match)
            getFeatures.fingerprint_process()
            getFeatures.detect_minutiae_pos_and_dir()
            getFeatures.create_local_structures()
            vm = getFeatures.valid_minutiae
            # print(vm)
            ls = getFeatures.local_structures
            # print(ls)
            fingerprint_for_compare, _ = qw.QFileDialog.getOpenFileName(self, os.getcwd())
            if fingerprint_for_compare:
                score = fc.fingerprint_comparison(vm, ls, fingerprint_for_compare)
                qw.QMessageBox.information(self, 'Match Result', f'The score is {score}')
    
    def evaluate_performance(self):
        self.matchName = self.ui.textEdit_Name.toPlainText()
        threshold = self.ui.textEdit_Threshold.toPlainText()

        try:
            if not threshold:
                threshold = 0.6
            else:
                threshold = float(threshold)
        except ValueError:
            qw.QMessageBox.warning(self, "Warning", "Threshold must be a number!")
            return
        
        if not self.matchName:
            qw.QMessageBox.warning(self, "Warning", "Name cannot be empty!")
        else :
            self.fingerprint_for_match, _ = qw.QFileDialog.getOpenFileName(self, os.getcwd())
            if self.fingerprint_for_match:  
                getFeatures = fp.Fingerprint(self.fingerprint_for_match)
                getFeatures.fingerprint_process()
                getFeatures.detect_minutiae_pos_and_dir()
                getFeatures.create_local_structures()
                self.vm = getFeatures.valid_minutiae
                self.ls = getFeatures.local_structures
                sample_folder = os.path.join(os.getcwd(), 'sample')
                npz_files = [f for f in os.listdir(sample_folder) if f.endswith('.npz')]
                
                matchedNamesCount = 0
                rightNameCount = 0
                results = []

                for npzFile in npz_files:
                    score = fc.fingerprint_comparison(self.vm, self.ls, npzFile)
                    getName = os.path.splitext(npzFile)[0]
                    name = getName.split('_')[0]

                    if score > threshold:
                        
                        matchedNamesCount += 1
                        results.append(f"{getName}: score: {score:.2f}")

                        if name == self.matchName:
                            rightNameCount += 1
                
                errorRate = 1 - rightNameCount / matchedNamesCount
                tempTuple = (errorRate, threshold)
                self.rocData.append(tempTuple)
                resultsInLines = "\n".join(results)
                self.ui.textEdit_Results.setPlainText(resultsInLines)
                
                qw.QMessageBox.information(self, 'Match Result', f'The error rate is {errorRate}')

    def draw_roc_curve(self):
        self.figure.clear()

        errorRates = [item[0] for item in self.rocData]
        thresholds = [item[1] for item in self.rocData]
        
        ax = self.figure.add_subplot(111)

        ax.plot(errorRates, thresholds, '-o', label='ROC Curve')
        ax.set_xlabel('Error Rate')
        ax.set_ylabel('Threshold')
        ax.set_title('ROC Curve')
        ax.legend()

        self.canvas.draw()

if __name__ == "__main__":
    app = qw.QApplication(sys.argv)
    w = RecognitionFrom()
    w.show()
    sys.exit(app.exec_())