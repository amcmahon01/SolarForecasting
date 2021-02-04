import sys
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
from pyqtgraph.dockarea import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import numpy as np
import netCDF4 as nc4


class FileWidget(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        hlay = QHBoxLayout(self)
        self.treeview = QTreeView()
        self.listview = QTreeView()
        hlay.addWidget(self.treeview)
        hlay.addWidget(self.listview)

        path = QDir.rootPath()

        self.dirModel = QFileSystemModel()
        self.dirModel.setRootPath(QDir.rootPath())
        self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllDirs)

        self.fileModel = QFileSystemModel()
        self.fileModel.setFilter(QDir.NoDotAndDotDot |  QDir.Files)

        self.treeview.setModel(self.dirModel)
        self.listview.setModel(self.fileModel)

        self.treeview.setRootIndex(self.dirModel.index(path))
        self.listview.setRootIndex(self.fileModel.index(path))

        self.treeview.setColumnWidth(0, 200)
        self.treeview.hideColumn(1)
        self.treeview.hideColumn(2)

        #Set to current path
        currentDir = self.dirModel.index(QDir.currentPath())
        self.treeview.setCurrentIndex(currentDir)
        self.on_clicked(currentDir)

        self.treeview.clicked.connect(self.on_clicked)
        #self.treeview.selectionModel().selectionChanged.connect(self.on_clicked)

    def setCallback(self, cb):
        helper = lambda index : cb(self.dirModel.fileInfo(index.indexes()[0]).absoluteFilePath())
        self.listview.selectionModel().selectionChanged.connect(helper)

    def on_clicked(self, index):
        path = self.dirModel.fileInfo(index).absoluteFilePath()
        self.listview.setRootIndex(self.fileModel.setRootPath(path))


class ImageArray(object):
    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return str(self.shape)


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.widgets = self.initUI()

    def initUI(self):
        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1200,800)
        self.setWindowTitle('Nowcasting netCDF Viewer')

        d1 = Dock("Files", size=(300, 600))     ## give this dock the minimum possible size
        d2 = Dock("Console", size=(600,50))
        d3 = Dock("Variables", size=(300,600))
        d4 = Dock("RGB Undistorted Image", size=(600,600))
        d5 = Dock("Red Mask ", size=(600,600))
        d6 = Dock("Bright Mask", size=(600,600))
        d7 = Dock("Cloud Mask", size=(600,600))
        area.addDock(d1, 'left')
        area.addDock(d2, 'bottom', d1)
        area.addDock(d3, 'bottom', d1)
        area.addDock(d4, 'right')
        area.addDock(d5, 'bottom', d4)
        area.addDock(d6, 'right', d4)
        area.addDock(d7, 'right', d5)

        ## first dock gets save/restore buttons
        w1 = FileWidget()
        w1.setCallback(self.loadNC)
        d1.addWidget(w1)

        self.con = pg.console.ConsoleWidget()
        self.con.catchAllExceptions(True)
        self.con.ui.exceptionGroup.setVisible(False)
        sys.stdout = self.con
        sys.stderr = self.con
        d2.addWidget(self.con)

        d = {}

        w3 = pg.DataTreeWidget(data=d)
        w3.setWindowTitle("Variables")
        d3.addWidget(w3)

        w4 = pg.ImageView()
        #w4.setImage(np.random.normal(size=(100,100)))
        #vbox = w4.getView()
        #vbox.addItem(pg.LabelItem("RGB Undistorted"))
        d4.addWidget(w4)

        w5 = pg.ImageView()
        #w5.setImage(np.random.normal(size=(100,100)))
        #vbox = w5.getView()
        #vbox.addItem(pg.LabelItem("Red Mask"))
        d5.addWidget(w5)

        w6 = pg.ImageView()
        #w6.setImage(np.random.normal(size=(100,100)))
        #vbox = w6.getView()
        #vbox.addItem(pg.LabelItem("Bright Mask"))
        d6.addWidget(w6)

        w7 = pg.ImageView()
        #w7.setImage(np.random.normal(size=(100,100)))
        #vbox = w7.getView()
        #vbox.addItem(pg.LabelItem("Cloud Mask"))
        d7.addWidget(w7)

        self.show()
        print("Ready!")

        w = {"vars": w3,"w_rgbu":w4,"w_red":w5,"w_br_mask":w6,"w_cl_mask":w7}
        return w

    def loadNC(self, f):
        print("Loading: " + os.path.basename(f))

        root_grp = nc4.Dataset(f, 'r', format='NETCDF4')
        d_vars = dict(root_grp.variables)
        d_out = {}
        for k, v in d_vars.items():
            if 'one' in v.dimensions:
                d_out.update({k: v[:][0]})      #scalar
            elif v.dimensions in (('x','y'),('x','y','z')):
                d_out.update({k: ImageArray(v.shape)})                   #images
            else:
                d_out.update({k: v[:]})

        self.widgets["vars"].setData(d_out)
        self.widgets["w_rgbu"].setImage(d_vars["RGBu"][:])
        self.widgets["w_red"].setImage(d_vars["Red"][:])
        self.widgets["w_br_mask"].setImage(d_vars["BrightMask"][:])
        self.widgets["w_cl_mask"].setImage(d_vars["CloudMask"][:])


if __name__ == '__main__':
    app = QtGui.QApplication([])
    viewer = Viewer()
    sys.exit(app.exec_())


