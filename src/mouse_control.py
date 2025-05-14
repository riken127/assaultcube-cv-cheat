# built-in library that allows us to have access to call functions and types from dlls and shared libraries.
import ctypes
# used to add delays
import time

# PUL is shorhand for `unsigned long` in c.
# used for the dwExtraInfo in the MOUSEINPUT win32 structure.
# this field can optionally hold additional information
# for the event being sent.
PUL = ctypes.POINTER(ctypes.c_ulong)

# This defines the MOUSEINPUT structure used by the Win32Api
# to describe mouse movement and events.
class MOUSEINPUT(ctypes.Structure):
    # dx and dy act as movement in the pixels along the X and Y axes.
    # mouse data adds aditional data depending on the event (e.g., wheel scrolls).
    # dwFlags acts as the a bit flag that allows us to specify the type of mouse event.
    # time is the timestamp for the event, zero lets the system set it automatically.
    # dwExtraInfo is application-specific value that be attached to the input.
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

# INPUT is the structure passed to the SendInput win32 function.
# This acts as a wrapper that can describe keyboard, mouse or hardware input.
# We used it to control mouse movements, even tho other movements are supported.
# union is used here because the windows INPUT structure behaves differently depending
# on its type field.
class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", ctypes.c_ulong), ("_input", _INPUT)]

INPUT_MOUSE = 0 # tells windows we are sending a mouse event.
MOUSEEVENTF_MOVE = 0x0001 # indicates movement of the mouse.
MOUSEEVENTF_LEFTDOWN = 0x0002 # indicates the left mouse button is being pressed.
MOUSEEVENTF_LEFTUP = 0x0004 # indicates that the left mouse button is being released.

# This function simulates moving the mouse cursor by `dx` and `dy` pixels from its current location
def move_mouse(dx, dy):
    # Creates a MOUSEINPUT instance to describe the mouse movement
    # sets mouseData to zero since we are not scrolling or sending extra data.
    # time=0, allows the OS to set the timestamp on itself.
    # dwExtraInfo=None since we aren't gonna specify any additional info
    mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=None)
    # Wraps the MOUSEINPUT structure into the INPUT structure with type INPUT_MOUSE
    inp = INPUT(type=INPUT_MOUSE, mi=mi)
    # SendInput is a win32 api function that synthesizes input events.
    # The first argument means we're sending one input event.
    # ctypes.pointer(inp) passes a pointer to our `INPUT`structure
    # ctypes.sizdeof(inp) tells the function how big our structure is.
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))

# This function simulates a complete left mouse click
def click_mouse():
    # Creates an extra information value to pass to the input event.
    # Not strictly needed but win32 expects a pointer
    extra = ctypes.c_ulong(0)
    
    # Creates an instance of the internal union class that wraps `MOUSEINPUT`
    ii_ = INPUT._INPUT()
    
    # Builds a `MOUSEINPUT` structure with no movement,
    # but with the left button being pressed (MOUSEEVENTF_LEFTDOWN)
    # Uses a pointer to the extra info value
    ii_.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
    command = INPUT(type=INPUT_MOUSE, _input=ii_)
    
    # Wraps it in an INPUT structure and sends it via SendInput, simulating a mouse button press
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
    
    # 10ms delay to mimic a human click
    time.sleep(0.01)
    
    # Updates the structure to simulate releasing the left mouse button.
    ii_.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))

    # Sends the release event to complete the click
    command = INPUT(type=INPUT_MOUSE, _input=ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))