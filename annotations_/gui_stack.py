# gui_stack.py
# Manages the video stack and ROI interactions for annotation.

from PyQt6.QtWidgets import QWidget, QGridLayout, QCheckBox, QComboBox, QLabel
from PyQt6.QtCore import Qt
from gui_imageview import ImageView  # Import custom ImageView
from gui_roi import ROI  # Import ROI class
import numpy as np
import imageio as io

class Stack(QWidget):
    def __init__(self, fn, rois=None):
        """
        Initialize the Stack widget for video annotation.
        
        Args:
            fn (str): Path to the video file.
            rois (list, optional): Pre-existing ROIs to load.
        """
        super().__init__()
        self.fn = fn
        self.colors = ['1a87f4', 'ebf441', '9b1a9b', '42f489']  # Colors for hexbugs
        self.curId = 0  # Current frame index
        self.freeze = False  # Prevent unnecessary updates

        # Load video frames into a numpy array
        self.im = np.asarray(io.mimread(self.fn, memtest=False))
        self.dim = self.im.shape  # Shape: (frames, height, width, channels)

        # Initialize ROIs for all frames
        self.rois = self.createROIs(rois)

        # Create ImageView for displaying frames
        self.w = ImageView(self.im.transpose(0, 2, 1, 3), parent=self)

        # Set up the grid layout
        self.l = QGridLayout()
        self.l.addWidget(self.w, 0, 0, 12, 1)

        # Combo box for selecting which hexbug to annotate
        self.annotate = QComboBox()
        self.annotate.addItems([f"Hexbug {i}" for i in range(1, 5)])
        self.annotate.currentIndexChanged.connect(self.changeAnnotating)
        self.annotating = 0  # Current hexbug being annotated

        # Checkboxes for showing/selecting hexbugs
        self.p1 = QCheckBox("show/select")
        self.p1.setStyleSheet(f"color: #{self.colors[0]}")
        self.p1.stateChanged.connect(self.checkROIs)
        self.p2 = QCheckBox("show/select")
        self.p2.setStyleSheet(f"color: #{self.colors[1]}")
        self.p2.stateChanged.connect(self.checkROIs)
        self.p3 = QCheckBox("show/select")
        self.p3.setStyleSheet(f"color: #{self.colors[2]}")
        self.p3.stateChanged.connect(self.checkROIs)
        self.p4 = QCheckBox("show/select")
        self.p4.setStyleSheet(f"color: #{self.colors[3]}")
        self.p4.stateChanged.connect(self.checkROIs)

        # Add UI elements to layout
        self.l.addWidget(QLabel("Hexbug 1"), 0, 1)
        self.l.addWidget(self.p1, 1, 1)
        self.l.addWidget(QLabel("Hexbug 2"), 2, 1)
        self.l.addWidget(self.p2, 3, 1)
        self.l.addWidget(QLabel("Hexbug 3"), 4, 1)
        self.l.addWidget(self.p3, 5, 1)
        self.l.addWidget(QLabel("Hexbug 4"), 6, 1)
        self.l.addWidget(self.p4, 7, 1)
        self.l.addWidget(QLabel("Currently annotating:"), 8, 1)
        self.l.addWidget(self.annotate, 9, 1)

        # Auto-move frame checkbox
        self.autoMove = QCheckBox("Automatically change frame")
        self.autoMove.setChecked(True)
        self.l.addWidget(self.autoMove, 10, 1)

        self.setLayout(self.l)

        # Initial setup
        self.updateCheckboxes()
        self.checkROIs()
        self.w.setROIs(self.rois[0])

    def createROIs(self, rois=None):
        """
        Create or load ROIs for each frame.
        
        Args:
            rois (list, optional): Pre-existing ROIs.
        
        Returns:
            list: List of ROI lists for each frame.
        """
        tmp_rois = [[ROI([100 + i * 25, 100 + i * 25], False) for i in range(4)]
                    for _ in range(self.dim[0])]
        if rois:
            for r in rois:
                tmp_rois[r['z']][r['id']].pos = r['pos']
                tmp_rois[r['z']][r['id']].shown = True
        return tmp_rois

    def updateCheckboxes(self):
        """Update checkbox states based on current frame's ROIs."""
        self.freeze = True
        self.p1.setChecked(self.rois[self.curId][0].shown)
        self.p2.setChecked(self.rois[self.curId][1].shown)
        self.p3.setChecked(self.rois[self.curId][2].shown)
        self.p4.setChecked(self.rois[self.curId][3].shown)
        self.freeze = False

    def checkROIs(self):
        """Show or hide ROIs based on checkbox states."""
        if not self.freeze:
            self.w.realRois[0].setVisible(self.p1.isChecked())
            self.w.realRois[1].setVisible(self.p2.isChecked())
            self.w.realRois[2].setVisible(self.p3.isChecked())
            self.w.realRois[3].setVisible(self.p4.isChecked())

    def changeAnnotating(self):
        """Update the currently annotated hexbug."""
        self.annotating = self.annotate.currentIndex()

    def mousePress(self, roi_id):
        """
        Handle mouse press to select an ROI.
        
        Args:
            roi_id (int): ID of the clicked ROI.
        """
        checkboxes = [self.p1, self.p2, self.p3, self.p4]
        checkboxes[roi_id].setChecked(True)
        if self.autoMove.isChecked():
            self.forceStep(1)

    def forceStep(self, direction=1):
        """
        Move to the next or previous frame.
        
        Args:
            direction (int): 1 for next, -1 for previous.
        """
        new_index = self.curId + direction
        if 0 <= new_index < self.dim[0]:
            self.w.setCurrentIndex(new_index)

    def changeZ(self, *args):
        """Handle frame change events."""
        self.rois[self.curId] = self.w.getROIs()
        self.curId = self.w.currentIndex
        self.updateCheckboxes()
        self.w.setROIs(self.rois[self.curId])