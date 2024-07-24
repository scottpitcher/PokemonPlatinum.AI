import subprocess
import time
import pyautogui

# Path to the DeSmuME executable, the ROM file, and desired state
# Won't be starting from scratch to avoid uneccessary cutscenes
desmume_executable = '/Applications/DeSmuME.app/Contents/MacOS/DeSmuME'
pokemon_rom = 'Platinum_Randomized.nds'
state_file = '/Users/scottpitcher/Library/Application Support/DeSmuME/0.9.13/States/Platinum_Randomized.ds4'
lua_script = 'env.lua'


# Start DeSmuME emulator
def open_emulator():
    subprocess.Popen([desmume_executable, pokemon_rom, '--load-lua-script', lua_script])
    time.sleep(5)

    pyautogui.hotkey('fn', 'F') # Full screen

    # Wait for the emulator to start (this is the amount of time until actions can be sent to emulator after loading screen)
    time.sleep(15) 
    
    # Load in the state
    key = f'4'
    time.sleep(0.2)
    pyautogui.press('x')
    time.sleep(0.2)
    pyautogui.press(key)
    print(f"Loaded state {key}")