
```markdown
# TRACO Annotation Tool

## Overview

The TRACO Annotation Tool is a standalone graphical user interface (GUI) application built with Python and PyQt6. It enables users to manually annotate the positions of up to four hexbugs in video frames by marking regions of interest (ROIs). The tool supports video playback, ROI management, trajectory visualization, and data export in JSON (`.traco`) or CSV formats. It is designed for creating labeled datasets for machine learning tasks, such as training models to predict hexbug positions.

## Features

- **Video Playback**: Load and navigate through video frames using a slider or keyboard shortcuts.
- **ROI Annotation**: Place draggable, color-coded ROIs for each hexbug with mouse clicks.
- **Trajectory Visualization**: Plot hexbug movements across frames using matplotlib.
- **Export Options**: Save annotations as JSON (`.traco`) or export to CSV for downstream use.
- **Configuration**: Uses a `settings.json` file to specify the default directory for file operations.
- **Keyboard Shortcuts**: Navigate frames (A/D), toggle ROIs (1-4), and save (Ctrl+S).

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed, along with the following libraries:
- `PyQt6`
- `pyqtgraph`
- `numpy`
- `imageio`
- `pandas`
- `matplotlib`

Install them via pip:
```bash
pip install PyQt6 pyqtgraph numpy imageio pandas matplotlib
```

### Project Setup

1. **Download the Code**: Obtain the source files, either as a single file (`traco_annotation_gui.py`) or as modular files (`gui_main.py`, `gui_stack.py`, etc.).
2. **Directory Structure**:
   ```
   project/
   ├── start.py                # Launcher script
   ├── traco_annotation_gui.py # Main script (if not modularized)
   ├── gui_main.py            # Modular main window (if modularized)
   ├── gui_stack.py           # Modular stack management
   ├── gui_imageview.py       # Modular image view
   ├── gui_roi.py             # Modular ROI class
   ├── gui_utils.py           # Modular utilities
   └── settings.json          # Configuration file
   ```
3. **Create `settings.json`** (if not auto-generated):
   ```json
   {
       "default_directory": "/path/to/your/video/directory"
   }
   ```
   Replace `"/path/to/your/video/directory"` with the path to your video files (e.g., `/home/user/videos` or `C:\Videos`).

## settings.json Details

The `settings.json` file configures the default directory for file dialogs (e.g., opening videos or saving annotations). It is a JSON file with the following structure:

```json
{
    "default_directory": "/path/to/your/video/directory"
}
```

- **Key**: `"default_directory"`
- **Value**: A string representing the absolute path to the directory where video files are stored or annotations are saved.
- **Behavior**:
  - If `settings.json` exists, the tool uses the specified `"default_directory"` for file dialogs.
  - If `settings.json` is missing, the tool creates it automatically, setting `"default_directory"` to the current working directory (obtained via `os.getcwd()`).
- **Customization**:
  - Edit `settings.json` to change the default directory to any valid path.
  - Ensure the directory exists to avoid file access errors.
- **Example**:
  ```json
  {
      "default_directory": "/home/user/videos"
  }
  ```

**Note**: Always use forward slashes (`/`) in the path, even on Windows, for compatibility. For example, use `"C:/Videos"` instead of `"C:\Videos"`.

## How to Use

Follow these steps to use the TRACO Annotation Tool:

1. **Prepare Your Environment**:
   - Ensure all dependencies are installed (see Prerequisites).
   - Place your video files (e.g., MP4 format) in a directory accessible to the tool.
   - Verify that `settings.json` exists in the project directory or let the tool create it automatically.

2. **Launch the Tool**:
   - Run the launcher script (`start.py`):
     ```bash
     python start.py
     ```

3. **Load a Video**:
   - From the "File" menu, select "Open".
   - Navigate to your video file (defaults to the directory specified in `settings.json`).
   - Supported format: MP4 (other formats may work depending on `imageio`).
   - If a corresponding `.traco` file exists (e.g., `video.traco` for `video.mp4`), existing annotations will load automatically.

4. **Annotate Hexbug Positions**:
   - **Navigate Frames**:
     - Use the slider in the GUI to move between frames.
     - Press `A` to go to the previous frame or `D` to go to the next frame.
   - **Select a Hexbug**:
     - Use the dropdown menu ("Currently annotating") to choose a hexbug (1-4).
   - **Place ROIs**:
     - Click on the video frame to place the ROI for the selected hexbug.
     - Each hexbug has a unique color (blue, yellow, purple, green).
   - **Show/Hide ROIs**:
     - Use the checkboxes labeled "show/select" to toggle visibility for each hexbug.
     - Press keys `1`, `2`, `3`, or `4` to toggle the corresponding hexbug’s ROI.
     - Press `Q` to show all ROIs.
   - **Automatic Frame Advance**:
     - Check "Automatically change frame" to advance to the next frame after placing an ROI.
   - **Adjust ROIs**:
     - Drag ROIs to fine-tune their positions.

5. **Save Annotations**:
   - Press `Ctrl+S` or select "Save" from the "File" menu.
   - Annotations are saved as a `.traco` file (JSON) in the same directory as the video (e.g., `video.traco`).
   - The `.traco` file contains frame numbers, hexbug IDs, and positions.

6. **Visualize Trajectories**:
   - Select "Plot trajectories" from the "Features" menu.
   - A matplotlib plot shows the first frame with overlaid hexbug paths (colored by hexbug ID).

7. **Export to CSV**:
   - Select "Export trajectories to CSV" from the "Features" menu.
   - Choose a save location and filename.
   - The CSV file contains columns: `t` (frame number), `hexbug` (ID), `x`, `y` (coordinates).

8. **Exit the Tool**:
   - Select "Exit" from the "File" menu.
   - Confirm to close, ensuring you’ve saved your annotations.
