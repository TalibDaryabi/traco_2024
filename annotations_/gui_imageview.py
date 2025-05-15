# gui_imageview.py
# Custom ImageView class for displaying video frames and handling ROIs.

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from gui_roi import ROI  # Import ROI class

class ImageView(pg.ImageView):
    keysignal = pyqtSignal(int)  # Signal for key presses
    mousesignal = pyqtSignal(int)  # Signal for mouse clicks

    def __init__(self, im, parent=None):
        """
        Initialize the ImageView for video display and interaction.
        
        Args:
            im (ndarray): Video frames as a numpy array.
            parent (QWidget, optional): Parent widget (Stack).
        """
        super().__init__(parent=parent)
        self.setImage(im)  # Set the video frames
        self.colors = ['#1a87f4', '#ebf441', '#9b1a9b', '#42f489']  # Hexbug colors
        self.realRois = []  # List of ROI objects

        # Create ROIs for each hexbug
        for i in range(4):
            t = pg.CrosshairROI([-1, -1])  # Initial position off-screen
            t.setPen(pg.mkPen(self.colors[i]))  # Set color
            t.aspectLocked = True  # Lock aspect ratio
            t.rotateAllowed = False  # Disable rotation
            self.realRois.append(t)
            self.getView().addItem(t)  # Add to view

        self.getView().setMenuEnabled(False)  # Disable right-click menu
        self.stack = parent  # Reference to parent Stack widget

    def mousePressEvent(self, e):
        """
        Handle mouse clicks to set ROI positions.
        
        Args:
            e (QMouseEvent): Mouse event data.
        """
        pos = e.pos()
        xy = self.getImageItem().mapFromScene(pos.x(), pos.y())  # Map to image coordinates
        if e.button() == Qt.MouseButton.LeftButton:
            self.realRois[self.stack.annotating].setPos(xy)  # Set position
            self.realRois[self.stack.annotating].show()  # Show ROI
            self.mousesignal.emit(self.stack.annotating)  # Emit signal

    def setROIs(self, rois):
        """
        Update ROI positions and visibility.
        
        Args:
            rois (list[ROI]): List of ROI objects for the current frame.
        """
        for i, r in enumerate(rois):
            self.realRois[i].setPos(r.pos)
            self.realRois[i].setVisible(r.shown)

    def getROIs(self):
        """
        Retrieve current ROI states.
        
        Returns:
            list[ROI]: List of ROI objects.
        """
        return [ROI(r.pos(), r.isVisible()) for r in self.realRois]

    def keyPressEvent(self, ev):
        """
        Emit key press events to parent.
        
        Args:
            ev (QKeyEvent): Key event data.
        """
        self.keysignal.emit(ev.key())