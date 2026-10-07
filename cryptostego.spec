# cryptostego.spec  —  PyInstaller build spec for CryptoStego
# Build command:
#   pyinstaller cryptostego.spec --clean
# Output:  dist/CryptoStego.exe  (standalone, no Python needed)

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Collect DLLs and data files for libraries that don't auto-collect
added_binaries = []
added_binaries += collect_dynamic_libs('cv2')
added_binaries += collect_dynamic_libs('PySide6')

added_datas = []
added_datas += collect_data_files('pydicom')      # DICOM UID tables etc.
added_datas += collect_data_files('sklearn')       # KMeans model data

a = Analysis(
    ['app_gui.py'],
    pathex=['.'],
    binaries=added_binaries,
    datas=added_datas,
    hiddenimports=[
        # OpenCV
        'cv2',
        # Scientific
        'numpy', 'scipy', 'scipy.io', 'scipy.io.wavfile',
        'sklearn', 'sklearn.cluster', 'sklearn.cluster._kmeans',
        'sklearn.utils._cython_blas',
        # DICOM
        'pydicom', 'pydicom.encoders', 'pydicom.pixel_data_handlers',
        'pydicom.pixel_data_handlers.numpy_handler',
        # PySide6 multimedia (Windows Media Foundation backend)
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # matplotlib (used by evaluation.py)
        'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_agg',
        # project modules
        'image_io', 'image_crypto', 'audio_stego', 'pipeline',
        'metrics', 'evaluation', 'chaotic_maps',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['streamlit', 'watchdog', 'pytest', 'tkinter', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CryptoStego',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,      # --windowed : no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,          # add an .ico file path here if desired
    onefile=True,       # --onefile : single portable .exe
)
