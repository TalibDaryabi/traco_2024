# gui_utils.py
# Utility functions for the TRACO Annotator GUI.

def trace():
    """Enable debugging with pdb in a PyQt environment."""
    from PyQt6.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace()