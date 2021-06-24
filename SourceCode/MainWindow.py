import sys, os
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QTabWidget, QAction, QVBoxLayout, QHBoxLayout, QLabel, 
                                QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QStyledItemDelegate, QAbstractItemView, QComboBox, QScrollArea,
                                    QListWidget, QListWidgetItem, QTextEdit, QProgressBar, QGridLayout, QSizePolicy, QMessageBox, QSpinBox)
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPen, QImage
from PyQt5.QtCore import Qt, QFile, QIODevice, QBasicTimer
import pyqtgraph as pg
import numpy as np
from torch.utils import data
from torch import nn, optim, cuda, no_grad, exp, save, load, Tensor, flatten
from torchvision import datasets, transforms
# Fix for HTTP error when trying to download MNIST; alternative mirror
new_mirror = 'http://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
   ('/'.join([new_mirror, url.split('/')[-1]]), md5)
   for url, md5 in datasets.MNIST.resources
]
import torch.nn.functional as F
import time
from PIL import Image
# Ensure 'resources' is installed in the environment

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        #determine the variable values
        self.title = "Handwriting Recognizer"
        self.status = "Training required..."
        self.left = 300
        self.top = 200
        self.width = 800
        self.height = 500
        self.setGeometry(300, 200, 800, 500)
        self.device = ''

        #set the window/program name and initial geometry
        self.setWindowTitle(self.title)
        self.statusBar().showMessage(self.status)
        self.setWindowIcon(QIcon('images/IK_logo.png'))
        self.setGeometry(self.left, self.top, self.width, self.height)

        #create the tabs widget and set it as the central widget
        self.tabsWidget = MyTabsWidget(self)
        self.setCentralWidget(self.tabsWidget)

        # Set an id for each iteration
        self.id = 0
        self.probs_list = []

        #display the application
        self.show()

    def closeEvent(self, event):
        # Delete created image file on exit
        try:
            for i in range(self.id):
                os.remove(f"images/i{i}.png")
        except:
            pass
        
        # Removes trained model on exit
        os.remove("mnist_model.pt")


class MyTabsWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        #create the layout for the mainWindow
        self.mainWindowlayout = QVBoxLayout(self)

        #initialize tabs
        self.tabs = QTabWidget()
        self.TrainTab = QWidget()
        self.ViewTab = QWidget()
        self.HistoryTab = QWidget()
        self.CanvasTab = QWidget() 

        # connects to onChange on change between tabs
        self.tabs.currentChanged.connect(self.onChange)
        
        #add the created tabs into the tabs widget
        self.tabs.addTab(self.TrainTab, 'Train')
        self.tabs.addTab(self.ViewTab, 'View')
        self.tabs.addTab(self.HistoryTab, 'History')
        self.tabs.addTab(self.CanvasTab, 'Canvas')

        #create the delegate object for readOnly
        self.readOnly = ReadOnlyDelegate(self)

        #create train tab (Kimsong Lor)
        self.createTrainTab()

        #create view tab (Isabelle Johns)
        self.createViewTab()

        #create history tab (Kimsong Lor)
        self.createHistoryTab()

        #create canvas tab (Isabelle Johns)
        self.createCanvasTab()

        #add the tabs to the tabs widget
        self.mainWindowlayout.addWidget(self.tabs)

        #set the layout for the mainWindow
        self.setLayout(self.mainWindowlayout)

    def loadMNIST(self):
        # Downloads and loads the MNIST dataset.
        batch_size = 10
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        print(f'Training MNIST Model on {self.device}\n{"=" * 44}')

        # Set train dataset
        self.train_dataset = datasets.MNIST(root='mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

        # Set test dataset
        self.test_dataset = datasets.MNIST(root='mnist_data/',
                                    train=False,
                                    transform=transforms.ToTensor())

        # Creates iterable for train data
        self.train_loader = data.DataLoader(dataset=self.train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

        # Creates iterable for train data
        self.test_loader = data.DataLoader(dataset=self.test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#---------------------TRAIN-----------------------#   
    #function to create train tab (Kimsong Lor)
    def createTrainTab(self):
        #create label and comboBox for model selection
        self.modelLabel = QLabel()
        self.modelLabel.setText('Model:')
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItems(['LeNet5', 'Linear Regression', 'Multilayer Perceptron'])

        #create label and comboBox for dataset selection
        self.datasetLabel = QLabel()
        self.datasetLabel.setText('Dataset:')
        self.datasetComboBox = QComboBox()
        self.datasetComboBox.addItems(['MNIST'])

        #create label and spinbox for amount of iterations
        self.iterationsLabel = QLabel()
        self.iterationsLabel.setText('Iterations:')
        self.iterationsSpinbox = QSpinBox()
        self.iterationsSpinbox.setRange(1, 100)

        #create text area for processsing statements
        self.trainingStatusText = QTextEdit()
        self.trainingStatusText.setFocusPolicy(Qt.NoFocus)

        #create loading bar
        self.trainingBar = QProgressBar(self)
        self.trainingBar.setAlignment(Qt.AlignCenter)
        self.trainingBar.setMaximumHeight(25)

        #create timestep
        self.step = 0

        #create buttons for train/cancel
        self.trainTrainButton = QPushButton()
        self.trainTrainButton.setText('Train')
        self.trainTrainButton.setMaximumSize(137, 30)

        #connecting train button to progress bar and training the model
        self.trainTrainButton.clicked.connect(self.trainModel)

        #create layouts and insert widgets
            #model selection vboxlayout
        self.TrainTab.modelSelection = QVBoxLayout(self)
        self.TrainTab.modelSelection.addWidget(self.modelLabel)
        self.TrainTab.modelSelection.addWidget(self.modelComboBox)

            #dataset selection vboxlayout
        self.TrainTab.datasetSelection = QVBoxLayout(self)
        self.TrainTab.datasetSelection.addWidget(self.datasetLabel)
        self.TrainTab.datasetSelection.addWidget(self.datasetComboBox)

            #Iterations selection vboxlayout
        self.TrainTab.iterationsSelection = QVBoxLayout(self)
        self.TrainTab.iterationsSelection.addWidget(self.iterationsLabel)
        self.TrainTab.iterationsSelection.addWidget(self.iterationsSpinbox)

            #model, dataset, iterations hboxlayout
        self.TrainTab.MDISelection = QHBoxLayout(self)
        self.TrainTab.MDISelection.addLayout(self.TrainTab.modelSelection)
        self.TrainTab.MDISelection.addLayout(self.TrainTab.datasetSelection)
        self.TrainTab.MDISelection.addLayout(self.TrainTab.iterationsSelection)

            #text area and loading bar vboxlayout
        self.TrainTab.textAndBar = QVBoxLayout(self)
        self.TrainTab.textAndBar.addWidget(self.trainingStatusText)
        self.TrainTab.textAndBar.addWidget(self.trainingBar)

            #buttons hboxlayout
        self.TrainTab.buttons = QHBoxLayout(self)
        self.TrainTab.buttons.setAlignment(Qt.AlignRight)
        self.TrainTab.buttons.addWidget(self.trainTrainButton)

            #train tab main vboxlayout
        self.TrainTab.layout = QVBoxLayout()
        self.TrainTab.layout.addLayout(self.TrainTab.MDISelection)
        self.TrainTab.layout.addLayout(self.TrainTab.textAndBar)
        self.TrainTab.layout.addLayout(self.TrainTab.buttons)
        self.TrainTab.setLayout(self.TrainTab.layout)
    
    #fuction to update the trainingStatusText
    def updateTrainingStatusText(self, msg):
        self.trainingStatusText.append(msg)

    #train the selected model with the selected dataset and iterations
    def trainModel(self):

        #checks for whether the train button has been pressed or the cancel button
        if (self.trainTrainButton.text() == 'Train'):

            #disable the trainTrainButton
            self.trainTrainButton.setEnabled(False)

            #update status to training in progress and change the buttontext
            self.parent().status = "Training in progress..." 
            self.parent().statusBar().showMessage(self.parent().status)
            self.updateTrainingStatusText("Training has started...")

            #get the selections chosen
            model = self.modelComboBox.currentText()
            dataset = self.datasetComboBox.currentText()
            iterations = self.iterationsSpinbox.value()
            
            #train the model
            self.model_run(model, iterations)
        else:
            return
        
        #re-enable trainTrainButton
        self.trainingBar.setValue(100)
        self.trainTrainButton.setEnabled(True)

    #timer for model training progress bar
    def trainingTimer(self, value, maxValue, prevMin):
        self.step = int((value/maxValue) * 100) + prevMin
        self.trainingBar.setValue(self.step)

    def model_run(self, dataset, iterations):
        # Model selection
        if dataset == 'LeNet5':
            model = LeNet5()
        elif dataset == 'Linear Regression':
            model = LinReg()
        elif dataset == 'Multilayer Perceptron':
            model = MLP()
        else:
            print("Something's gone wrong!")

        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        def train(epoch, maxRange):
            model.train()
            prevMin = int(((epoch - 1)/maxRange) * 100)
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                #update timer test
                self.trainingTimer(batch_idx, int((len(self.train_loader) - 1)*maxRange), prevMin)

                if batch_idx % 10 == 0:
                    self.updateTrainingStatusText('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))
        
        def test():
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                # sum up batch loss
                test_loss += criterion(output, target).item()
                # get the index of the max
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(self.test_loader.dataset)
            self.updateTrainingStatusText(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} '
                f'({100. * correct / len(self.test_loader.dataset):.0f}%)')

        since = time.time()
        self.trainingBar.setValue(0)
        for epoch in range(1, int(iterations) + 1):
            #set timer for progress bar to zero
            epoch_start = time.time()
            train(epoch, int(iterations))
            m, s = divmod(time.time() - epoch_start, 60)
            self.updateTrainingStatusText(f'Training time: {m:.0f}m {s:.0f}s')
            test()
            m, s = divmod(time.time() - epoch_start, 60)
            self.updateTrainingStatusText(f'Testing time: {m:.0f}m {s:.0f}s')

        m, s = divmod(time.time() - since, 60)
        self.updateTrainingStatusText(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {self.device}!')

        save(model, './mnist_model.pt')
        #update status after model has finished training
        self.parent().status = "Ready!" 
        self.parent().statusBar().showMessage(self.parent().status)
        self.updateTrainingStatusText("Training has been complete!")
        self.step = 0
        self.trainTrainButton.setText('Train')
    
#-------------------------------------------------#


#----------------------VIEW-----------------------#
    #function to create view tab (Isabelle Johns)
    def createViewTab(self):
        self.loadMNIST()

        # Dataset selection combobox
        self.set_comboBox = QComboBox()
        self.set_comboBox.addItems(["Training", "Testing"])
        self.set_label = QLabel()
        self.set_label.setText("Image set:")
        
        # Filter combobox
        self.filter_comboBox = QComboBox()
        self.filter_comboBox.addItems(["None", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        self.filter_label = QLabel()
        self.filter_label.setText("Filter by digit:")

        # Apply filters button
        self.view_btn = QPushButton("Apply Filters >>")
        self.view_btn.clicked.connect(self.applyFilters)
        self.filter_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.view_btn.setMinimumHeight(55)

        # Create scrollarea
        self.scroll_area = QScrollArea(self)

        # Create grid layout for viewer
        self.view_grid = QGridLayout(self)
        self.view_grid.setContentsMargins(25, 25, 0, 25)
        self.view_grid.setSpacing(50)
        
        # Set initial page number to the first page
        self.page_num = 0

        # Navigation buttons
        self.prev_page_btn = QPushButton("<< Prev")
        self.prev_page_btn.setEnabled(False)
        self.prev_page_btn.clicked.connect(self.prevPage)

        self.next_page_btn = QPushButton("Next >>")
        self.next_page_btn.clicked.connect(self.nextPage)

        self.nav_label = QLabel()
        self.nav_label.setAlignment(Qt.AlignCenter)

        # Initialise dataset to training
        self.current_dataset = self.train_dataset
        self.length_of_selection = len(self.current_dataset)

        # Setting layout of tab --->
        self.view_area = QWidget()
        self.view_area.setLayout(self.view_grid)
        self.scroll_area.setWidget(self.view_area)
        
        self.ViewTab.setBox = QVBoxLayout(self)
        self.ViewTab.setBox.addWidget(self.set_label)
        self.ViewTab.setBox.addWidget(self.set_comboBox)
        
        self.ViewTab.filterBox = QVBoxLayout(self)
        self.ViewTab.filterBox.addWidget(self.filter_label)
        self.ViewTab.filterBox.addWidget(self.filter_comboBox)
        
        self.ViewTab.topBox = QHBoxLayout(self)
        self.ViewTab.topBox.addLayout(self.ViewTab.setBox)
        self.ViewTab.topBox.addLayout(self.ViewTab.filterBox)
        self.ViewTab.topBox.addWidget(self.view_btn)

        self.ViewTab.navBox = QHBoxLayout(self)
        self.ViewTab.navBox.addWidget(self.prev_page_btn)
        self.ViewTab.navBox.addWidget(self.nav_label)
        self.ViewTab.navBox.addWidget(self.next_page_btn)

        self.ViewTab.layout = QVBoxLayout(self)
        self.ViewTab.layout.addLayout(self.ViewTab.topBox)
        self.ViewTab.layout.addWidget(self.scroll_area)
        self.ViewTab.layout.addLayout(self.ViewTab.navBox)

        self.ViewTab.setLayout(self.ViewTab.layout)
        # <---


    def resizeEvent(self, event): # detect change in scrollArea size
        self.scroll_area_columns = int(self.scroll_area.geometry().width() / 80) # Set amount of columns in the image viewer
        self.redrawViewer()


    def onChange(self): # updates change in scrollArea size on change of tab
        self.scroll_area_columns = int(self.scroll_area.geometry().width() / 80) # Set amount of columns in the image viewer
        self.redrawViewer()


    def prevPage(self):
        # View previous 200 images in dataset
        self.page_num -= 1
        self.redrawViewer()
        if not self.next_page_btn.isEnabled():
            self.next_page_btn.setEnabled(True)
        if self.page_num == 0:
            self.prev_page_btn.setEnabled(False)


    def nextPage(self):
        # View next 200 images in dataset
        self.page_num += 1
        self.redrawViewer()
        if not self.prev_page_btn.isEnabled():
            self.prev_page_btn.setEnabled(True)
        if self.page_num == int(self.length_of_selection/200):
            self.next_page_btn.setEnabled(False)


    def applyFilters(self):
        
        # Set dataset to filter
        if self.set_comboBox.currentText() == 'Training':
            self.current_dataset = self.train_dataset
        else:
            self.current_dataset = self.test_dataset

        # Set digit to filter
        if self.filter_comboBox.currentText() == 'None':
            self.filter_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.length_of_selection = len(self.current_dataset)
        else:
            self.filter_value = [int(self.filter_comboBox.currentText())]

            # Count instances of filter digit
            count = 0
            for i in range(len(self.current_dataset)):
                if self.current_dataset.targets[i] == int(self.filter_value[0]):
                    count += 1
            self.length_of_selection = count

        # Reset page number to 0
        self.page_num = 0
        self.prev_page_btn.setEnabled(False)
        self.next_page_btn.setEnabled(True)

        # Update viewer
        self.redrawViewer()


    def redrawViewer(self):
        # Delete all images from scrollArea widget
        for i in reversed(range(self.view_grid.count())): 
            self.view_grid.itemAt(i).widget().setParent(None)
                                                                                                                                                                              
        count = 0
        amount_printed = 0

        # View 200 images, unless there aren't 200 images left to view; if not, just view remaining images
        if (self.length_of_selection - self.page_num*200) < 200:
            amount_to_display = self.length_of_selection % 200
        else:
            amount_to_display = 200

        # Displays 200 images with number of columns dependent on dynamic resizing of window
        for row in range(int(200/self.scroll_area_columns)+1):
            for column in range(self.scroll_area_columns):
                if amount_printed < amount_to_display: # self.current_dataset.targets[count] gets the label of the image
                    # x is a torch.Tensor
                    self.x, _ = self.current_dataset[200*self.page_num + count]  
                    
                    while self.current_dataset.targets[200*self.page_num + count] not in self.filter_value:
                        count += 1
                        self.x, _ = self.current_dataset[200*self.page_num + count] 

                    # Convert tensor to numpy array of 8-bit unsigned int values (0 - 255)
                    x_img = (255-(self.x.numpy()*255)).astype(np.uint8)

                    # Array we want is the first element of the numpy array (second element is type)
                    x_img = x_img[0]

                    # Creates image - takes in data, width of img, height of img, and format
                    qimage = QImage(x_img, x_img.shape[0], x_img.shape[1], QImage.Format_Grayscale8)                                                                                                                                                               
                    
                    # Creates pixmap of image
                    pixmap = QPixmap(qimage) 
                    
                    # Place image in layout
                    self.img_lbl = QLabel(self)
                    self.img_lbl.setPixmap(pixmap)
                    self.view_grid.addWidget(self.img_lbl, row, column)
                    amount_printed += 1
                    count += 1
                else:
                    pass
        
        # Set label on viewer
        self.nav_label.setText("Viewing {}-{} of {} images".format((self.page_num*200) + 1, (self.page_num*200) + amount_to_display, self.length_of_selection))
        self.view_area = QWidget()
        self.view_area.setLayout(self.view_grid)
        self.scroll_area.setWidget(self.view_area)
#-------------------------------------------------#



#--------------------HISTORY----------------------#
    #history tab creation (Kimsong Lor)
    def createHistoryTab(self):
        #create the history tab layout
        self.HistoryTab.layout = QVBoxLayout(self)

        #create table widget
        self.HistoryTableWidget = QTableWidget()

        #create horizontal header labels
        headerLabels = ['ID', 'Model', 'Dataset', 'Iterations', 'Recognition', 'Probablilty']

        #set amount of columns
        self.HistoryTableWidget.setColumnCount(len(headerLabels))

        #set vertical/horizontal header labels
        self.HistoryTableWidget.verticalHeader().setVisible(False)
        self.HistoryTableWidget.setHorizontalHeaderLabels(headerLabels)

        #set selection behavior to select entire row
        self.HistoryTableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        #fit the table to the screen horizontally
        self.HistoryTableWidget.horizontalHeader().setStretchLastSection(True)
        self.HistoryTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        #add the table widget to the history tab
        self.HistoryTab.layout.addWidget(self.HistoryTableWidget)

        #create button for viewing and link it to get data for selected listing
        self.HistoryViewButton = QPushButton("View")
        self.HistoryViewButton.setMaximumSize(146, 30)
        self.HistoryViewButton.clicked.connect(self.displaySelectedListing)

        #create horizontal layout for buttons
        self.HistoryTab.buttons = QHBoxLayout(self)
        self.HistoryTab.buttons.setAlignment(Qt.AlignRight)

        #add button to button layout and add that layout to the historytab layout
        self.HistoryTab.buttons.addWidget(self.HistoryViewButton)
        self.HistoryTab.layout.addLayout(self.HistoryTab.buttons)

        #set the history tab layout
        self.HistoryTab.setLayout(self.HistoryTab.layout)
    
    #adds row of data to history list
    def addDataHistoryList(self, table, rowData):
        #insert row at the top of the list
        table.insertRow(0)

        #add the data into the inserted row
        col = 0
        for item in rowData:
            data = QTableWidgetItem(str(item))
            data.setTextAlignment(Qt.AlignCenter)
            table.setItem(0, col, data)
            col += 1

        #make all inserted rows readOnly
        for i in range (0, table.rowCount()):
            table.setItemDelegateForRow(i, self.readOnly)

    #get data from selected listing
    def displaySelectedListing(self):
        #determine which listing was selected
        id = int(self.HistoryTableWidget.selectedItems()[0].text())

        #get the canvas drawing and probablilty chart
        self.canvasImage = QPixmap(f'images/i{id}.png')
        self.canvasImage = self.canvasImage.scaled(280, 280, Qt.KeepAspectRatio)
        # print(f'i{id}.png')
        self.probList = self.parent().probs_list[id]
        self.plotNew = pg.BarGraphItem(x = np.arange(10), height=self.probList, width = 0.6, brush="#5f76a3")

        #create the second window, pass in the graphs, then display
        self.popUp = historyDataPopUp()
        self.passingGraphs()
        self.popUp.displayGraphs()

    #insert graphs into the second window
    def passingGraphs(self):
        self.popUp.graph1.clear()
        self.popUp.graph1.setPixmap(self.canvasImage)
        self.popUp.graph1.resize(self.canvasImage.width(),self.canvasImage.height())
        self.popUp.graph2.removeItem(self.plot)
        self.popUp.graph2.addItem(self.plotNew)
        
#-------------------------------------------------#



#--------------------CANVAS-----------------------#
    #function to create canvas tab (Isabelle Johns)
    def createCanvasTab(self):
        # Create buttons
        self.clear_button = QPushButton("Clear")
        self.recog_button = QPushButton("Recognise")

        # Max size of buttons = size of canvas 
        self.clear_button.setMaximumSize(137, 30)
        self.recog_button.setMaximumSize(137, 30)

        # Create canvas
        self.c = QLabel()
        self.canvas = QPixmap(280, 280)
        self.canvas.fill(QColor("white"))
        self.c.setPixmap(self.canvas)
        self.draw_border()
        
        # Events activated on click of canvas
        self.c.mousePressEvent = self.begin_drawing
        self.c.mouseMoveEvent = self.user_draw
        self.c.mouseReleaseEvent = self.finish_drawing

        # Initialise painter's 'last' coordinates 
        self.last_x = None
        self.last_y = None
        
        self.clear_button.clicked.connect(self.clear_canvas)
        self.recog_button.clicked.connect(self.predict)

        # Graph creation
        self.g = pg.PlotWidget()
        self.g.setMouseEnabled(x=False, y=False) # Disable panning
        self.g.hideButtons()

        self.g.setLabel(axis='left', text='Probability')
        self.g.setLabel(axis='bottom', text='Digit')

        self.g.setYRange(0, 1)
        self.g.setXRange(0, 9)
        self.g.setBackground('w')

        # X ranges from 0-9
        x = np.arange(10)
        # No values of y initially
        y1 = 0
        
        self.plot = pg.BarGraphItem(x=x, height=y1, width=0.6, brush="#5f76a3")
        self.g.addItem(self.plot)

        # Prediction label
        self.predict_val = QLabel()
        self.predict_val.setAlignment(Qt.AlignCenter)

        # Horizontal layout of buttons
        self.CanvasTab.buttons = QHBoxLayout(self)
        self.CanvasTab.buttons.addWidget(self.clear_button)
        self.CanvasTab.buttons.addWidget(self.recog_button)

        # Vertical layout of canvas and buttons
        self.CanvasTab.leftSide = QVBoxLayout(self)
        self.CanvasTab.leftSide.addWidget(self.c)
        self.CanvasTab.leftSide.addLayout(self.CanvasTab.buttons)

        # Veritical layout of graph and prediction label
        self.CanvasTab.rightSide = QVBoxLayout(self)
        self.CanvasTab.rightSide.addWidget(self.g)
        self.CanvasTab.rightSide.addWidget(self.predict_val)

        # Horizontal layout of canvas and graph
        self.CanvasTab.layout = QHBoxLayout(self)
        self.CanvasTab.layout.addLayout(self.CanvasTab.leftSide)
        self.CanvasTab.layout.addLayout(self.CanvasTab.rightSide)

        # Set overall layout
        self.CanvasTab.setLayout(self.CanvasTab.layout)

    def draw_border(self):
        # Draw a 1px black border (to differentiate canvas from background)
        painter = QPainter(self.c.pixmap())
        painter.drawRect(0, 0, 279, 279)
        painter.end()

    def clear_canvas(self):
        # Draw a white rectangle over the entire canvas, then redraw the border
        painter = QPainter(self.c.pixmap())
        painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))

        painter.drawRect(0, 0, 279, 279)
        painter.setBrush(QBrush(Qt.NoBrush)) # Reinitialise brush so that there's no fill in future drawings

        painter.end()
        self.draw_border()
        self.update()

    def save_canvas(self):
        # Manually remove black border
        painter = QPainter(self.c.pixmap())
        painter.setPen(QColor("white"))
        painter.drawRect(0, 0, 279, 279)
        painter.end()

        # Create a blank 28x28 canvas
        canvas_28 = QPixmap(28, 28)
        canvas_28.fill(QColor("white"))

        # Paste 20x20 user input on top of 28x28 canvas
        painter = QPainter(canvas_28)
        painter.drawPixmap(4, 4, 20, 20, self.c.pixmap())
        painter.end()

        # Turn image to be saved into label
        self.input_scaled = QLabel()
        self.input_scaled.setPixmap(canvas_28)

        # Save label to file 'images'
        image = QFile(f"images/i{self.parent().id}.png")
        image.open(QIODevice.WriteOnly)
        result = self.input_scaled.pixmap().save(image, "PNG")

        # Redraw the border on the display
        self.draw_border()

    def predict(self):
        # Check that saved model exists
        if not os.path.isfile('mnist_model.pt'):
            err = QMessageBox()
            err.setIcon(QMessageBox.Information)
            err.setText("No model trained!\nPlease train one in the 'Train' tab before proceeding.")
            err.setStandardButtons(QMessageBox.Ok)
            err.show()
            err.exec_()
            return

        # Save the user's drawn digit
        self.save_canvas()

        # Check if image the user is trying to save matches the last image saved; if so, don't save a duplicate
        if self.parent().id > 0 and open(f'images/i{self.parent().id-1}.png', 'rb').read() == open(f'images/i{self.parent().id}.png', 'rb').read():
            os.remove(f'images/i{self.parent().id}.png')

        else:
            # Open the image and convert it to a tensor, values ranged from 0-1
            user_input = Image.open(f'images/i{self.parent().id}.png').convert('L')
            user_input = 1 - (np.array(user_input)/255)
            user_input = Tensor(user_input.reshape(-1, 1, 28, 28))

            # Load the saved trained model
            model = load('mnist_model.pt')

            # Feed image into trained model
            with no_grad():
                ps = model(user_input)
            probs = list(ps.numpy()[0])
            
            # Scales the probabilities to decimals between 0 and 1
            tot_probs = 0
            for i in range(len(probs)): # Gets sum of all probabilities
                tot_probs += probs[i]
            scaled_probs = probs/tot_probs # Divide each probability by the total
            print(scaled_probs)
            self.parent().probs_list.append(scaled_probs) 

            # Plot the probabilities
            self.g.removeItem(self.plot)
            self.plot = pg.BarGraphItem(x = np.arange(10), height=scaled_probs, width = 0.6, brush="#5f76a3")
            self.g.addItem(self.plot)

            #add listing to the history tab
            self.addDataHistoryList(self.HistoryTableWidget, [self.parent().id, self.modelComboBox.currentText(), self.datasetComboBox.currentText() , self.iterationsSpinbox.value(), probs.index(max(probs)), scaled_probs[probs.index(max(probs))]]) #testing data

            # Increase ID value by one for next prediction
            self.parent().id += 1

            # Display to user the predicated value
            self.predict_val.setText(f"Your input has been recognised as <b>{probs.index(max(probs))}</b>.")
            

    def begin_drawing(self, event):
        # Set the 'last' coordinate to the point where the mouse first clicked
        self.last_x = event.x()
        self.last_y = event.y()

        # Set the 280x280 pixmap on the canvas tab as the canvas
        painter = QPainter(self.c.pixmap())
        p = painter.pen()
        p.setWidth(28)
        p.setCapStyle(Qt.RoundCap)
        p.setColor(QColor("black"))
        painter.setPen(p)

        # Canvas coordinates do not match mouse coordinates; scale accordingly
        scaled_y = int(-(self.c.height()/2)+138)

        # Draw a point at clicked coordinate
        painter.drawPoint(self.last_x, self.last_y+scaled_y)
        painter.end()
        self.update()

    def user_draw(self, event):
        # Set canvas and pen
        painter = QPainter(self.c.pixmap())
        p = painter.pen()
        p.setWidth(28)
        p.setCapStyle(Qt.RoundCap)
        p.setColor(QColor("black"))
        painter.setPen(p)

        # Scaling so that canvas continues to operate properly on resize
        scaled_y = int(-(self.c.height()/2)+138)    

        # Draw a line between the last polled point and the current polled point
        painter.drawLine(self.last_x, self.last_y+scaled_y, event.x(), event.y()+scaled_y)
        painter.end()
        self.update()

        # Update the origin for next time
        self.last_x = event.x()
        self.last_y = event.y()

    def finish_drawing(self, event):
        # Reinitialise 'last' coordinate, ready for next drawing input
        self.last_x = None
        self.last_y = None
#-------------------------------------------------#


#--------------------LeNet5-----------------------#
class LeNet5(nn.Module): 

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            # 3 convolution layers
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
            nn.Tanh()
        )
        self.fc = nn.Sequential(
            # Two fully connected layers 
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )


    def forward(self, x):
        
        x = self.conv1(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return F.softmax(x, dim=1)
#-------------------------------------------------#



#---------------LINEAR REGRESSION-----------------#

class LinReg(nn.Module):
    def __init__(self):
        super(LinReg, self).__init__()
        # One linear transformation
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = flatten(x, 1)
        x = self.linear(x)
        return exp(x)

#-------------------------------------------------#



#--------------MULTILAYER PERCEPTRON--------------#

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            # A linear transformation, activation layer then an output layer
            nn.Linear(784, 120),
            nn.ReLU(),
            nn.Linear(120, 10)
        )

    def forward(self, x):
        x = flatten(x, 1)
        x = self.layers(x)
        return exp(x)

#-------------------------------------------------#



#-----------------POPUP WINDOW--------------------#
#pop up window to display old data
class historyDataPopUp(QWidget):
    def __init__(self):
        super().__init__()

        #create widget for the popUp window
        self.popUpWindow = QWidget()

        #set geometry
        self.popUpWindow.setGeometry(500, 300, 700, 400)
        self.popUpWindow.setWindowTitle("Preview")

        #graph (canvas)
        self.graph1 = QLabel()
        ex_img = QPixmap('images/example_image.png')
        self.graph1.setPixmap(ex_img)
        self.graph1.resize(ex_img.width(),ex_img.height())

        #graph (probablilty chart)
        self.graph2 = pg.PlotWidget()
        self.graph2.setMouseEnabled(x=False, y=False) # Disable panning
        self.graph2.hideButtons()

        self.graph2.setLabel(axis='left', text='Probability')
        self.graph2.setLabel(axis='bottom', text='Digit')

        self.graph2.setYRange(0, 1)
        self.graph2.setXRange(0, 9)
        self.graph2.setBackground('w')

        x = np.arange(10)
        y1 = 0
        
        self.plot = pg.BarGraphItem(x=x, height=y1, width=0.6, brush="#5f76a3")
        self.graph2.addItem(self.plot)

        #make horizontal layout
        self.popUpWindow.layout = QHBoxLayout()
        self.popUpWindow.layout.addWidget(self.graph1)
        self.popUpWindow.layout.addWidget(self.graph2)

        #set layout
        self.popUpWindow.setLayout(self.popUpWindow.layout)

    def displayGraphs(self):   
        self.popUpWindow.show()
#-------------------------------------------------#


#---------------READONLY DELEGATE-----------------#
#read only delegate
class ReadOnlyDelegate(QStyledItemDelegate):
    #when an editor is created, do nothing
    def createEditor(self, parent, option, index):
        return
#-------------------------------------------------#


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())