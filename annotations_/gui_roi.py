# gui_roi.py
# ROI class for managing position and visibility of regions of interest.

from pyqtgraph import Point

class ROI:
    def __init__(self, pos=[100, 100], shown=True):
        """
        Initialize an ROI object.
        
        Args:
            pos (list or tuple or Point): Initial position.
            shown (bool): Visibility flag.
        """
        self.pos = pos if isinstance(pos, Point) else Point(pos)
        self.shown = shown

    def serialize(self):
        """
        Convert ROI to a serializable dictionary.
        
        Returns:
            dict: ROI data with position and visibility.
        """
        return {'pos': tuple(self.pos), 'shown': self.shown}