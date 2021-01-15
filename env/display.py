import os
import warnings
from pyvirtualdisplay import Display

try:
    ORIGIN_DISP = os.environ['DISPLAY']
except KeyError:
    warnings.warn('''You may be run the process in docker which has no X server.
     We may set the DISPLAY env to an empty str for avoiding exception''')
    ORIGIN_DISP = ''  # Docker in case
DISP = Display(visible=False, size=(1400, 900))
DISP.start()
VIRTUAL_DISP = os.environ['DISPLAY']
os.environ['DISPLAY'] = ORIGIN_DISP


def virtual_display(func):
    def wrapper(*args, **kwargs):
        os.environ['DISPLAY'] = VIRTUAL_DISP
        func(*args, **kwargs)
        os.environ['DISPLAY'] = ORIGIN_DISP
    return wrapper
