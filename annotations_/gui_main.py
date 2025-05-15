# gui_main.py
# Main window class for the TRACO Annotator GUI, handling file operations and menu actions.

import json
import os
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtCore import Qt
from gui_stack import Stack  # Import Stack class from gui_stack.py
import pandas as pd
import matplotlib.pyplot as plt

class Main(QMainWindow):
    def __init__(self):
        """Initialize the TRACO Annotator main window."""
        super().__init__()
        self.settings_fn = None  # Path to settings file (not used currently)
        self.status = self.statusBar()  # Status bar for displaying messages
        self.menu = self.menuBar()  # Menu bar for file and feature actions

        # File menu setup
        self.file = self.menu.addMenu("&File")
        self.file.addAction("Open", self.open)  # Open video file
        self.file.addAction("Save", self.save)  # Save annotations
        self.file.addAction("Exit", self.close)  # Exit application

        # Features menu setup
        self.features = self.menu.addMenu("&Features")
        self.features.addAction("Plot trajectories", self.plotTrajectories)  # Plot hexbug paths
        self.features.addAction("Export trajectories to CSV", self.export)  # Export to CSV

        self.fn = None  # Path to the current video file
        self.history = []  # Unused list for history tracking

        # Set window properties
        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle("TRACO Annotator")

    def plotTrajectories(self):
        """Plot hexbug trajectories using matplotlib."""
        if not self.fn:
            return  # Exit if no video is loaded

        plt.figure()
        plt.imshow(self.stack.im[0])  # Display first frame as background

        # Prepare time steps and positions for each hexbug
        ts = [[] for _ in range(4)]
        xys = [[] for _ in range(4)]

        # Collect positions from all frames
        for i in range(self.stack.dim[0]):  # Iterate over frames
            for j in range(4):  # Iterate over hexbugs
                r = self.stack.rois[i][j]
                if r.shown:
                    ts[j].append(i)
                    xys[j].append(r.pos)

        # Plot trajectories with unique colors
        for i, xy in enumerate(xys):
            for j in xy:
                plt.scatter(*j, color="#" + self.stack.colors[i])

        plt.xlim([0, self.stack.dim[2]])
        plt.ylim([self.stack.dim[1], 0])
        plt.show()

    def export(self):
        """Export annotated hexbug positions to a CSV file."""
        with open('settings.json', 'r') as file:
            keys_2_settings = json.load(file)

        # Prompt user for save location
        fn = QFileDialog.getSaveFileName(directory=keys_2_settings["default_directory"], filter="*.csv")[0]
        if fn:
            tmp = []
            # Gather data for all hexbugs across frames
            for j in range(4):  # Hexbug IDs
                for i in range(len(self.stack.rois)):  # Frames
                    r = self.stack.rois[i][j]
                    if r.shown:
                        e = {'t': i, 'hexbug': j, 'x': r.pos[1], 'y': r.pos[0]}
                        tmp.append(e)
            pd.DataFrame(tmp).to_csv(fn)  # Save to CSV
            QMessageBox.information(self, "Data exported.", f"Data saved at\n{fn}")

    def close(self):
        """Close the application with confirmation."""
        ok = QMessageBox.question(self, "Exiting?", "Do you really want to exit? Ensure you save your progress.")
        if ok == QMessageBox.Yes:
            super().close()

    def open(self):
        """Open a video file and initialize the annotation interface."""
        with open('settings.json', 'r') as file:
            keys_2_settings = json.load(file)

        # Prompt user to select a video file
        fn = QFileDialog.getOpenFileName(directory=keys_2_settings["default_directory"])[0]
        if fn:
            self.fn = fn
            self.fn_rois = fn.replace("mp4", "traco")  # Annotation file path

            # Load existing annotations if available
            if os.path.isfile(self.fn_rois):
                with open(self.fn_rois, 'r') as fp:
                    rois = json.load(fp)['rois']
            else:
                rois = None

            # Initialize the Stack widget with video and ROIs
            self.stack = Stack(self.fn, rois=rois)
            self.setCentralWidget(self.stack)

            # Connect signals for status updates and keyboard shortcuts
            self.stack.w.sigTimeChanged.connect(self.updateStatus)
            self.stack.w.keysignal.connect(self.savekeyboard)

            self.setWindowTitle(f"TRACO Annotator | Working on file {self.fn}")

    def updateStatus(self):
        """Update the status bar with current frame and dimensions."""
        self.status.showMessage(f"z: {self.stack.w.currentIndex} x: {self.stack.dim[0]} y: {self.stack.dim[1]}")

    def save(self):
        """Save current annotations to a JSON file."""
        if self.fn_rois:
            with open(self.fn_rois, "w") as fp:
                json.dump({
                    "rois": [{'z': i, 'id': j, 'pos': self.stack.rois[i][j].serialize()['pos']}
                             for i in range(len(self.stack.rois))
                             for j in range(len(self.stack.rois[i]))
                             if self.stack.rois[i][j].shown]
                }, fp, indent=4)
            self.status.showMessage(f"ROIs saved to {self.fn_rois}", 1000)

    def savekeyboard(self, key):
        """Handle Ctrl+S shortcut to save annotations."""
        from PyQt6.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        if key == Qt.Key.Key_S and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.save()