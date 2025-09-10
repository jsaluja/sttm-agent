#!/usr/bin/env python3

import os
import time
import subprocess
import logging
import re
from PIL import ImageGrab
import pytesseract

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed coordinates for phone icon
PHONE_ICON_X = 25
PHONE_ICON_Y = 785

def activate_sttm():
    """
    Activate the STTM application window (bring to front) or open it if not running
    """
    try:
        logger.info("Activating STTM application window")
        
        # First check if SikhiToTheMax is running
        check_script = '''
        tell application "System Events"
            return (exists process "SikhiToTheMax")
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', check_script], capture_output=True, text=True)
        app_running = result.stdout.strip().lower() == 'true'
        
        if not app_running:
            logger.info("STTM is not running. Attempting to open it...")
            # Try to open the app - first attempt using AppleScript
            open_script = '''
            tell application "SikhiToTheMax" to activate
            '''
            subprocess.run(['osascript', '-e', open_script], capture_output=True)
            time.sleep(3)  # Wait for app to launch
            
            # Check if it opened successfully
            result = subprocess.run(['osascript', '-e', check_script], capture_output=True, text=True)
            app_running = result.stdout.strip().lower() == 'true'
            
            if not app_running:
                # Try alternative methods to open the app
                logger.info("First attempt failed. Trying alternative methods...")
                
                # Try to find the app in standard locations
                app_paths = [
                    "/Applications/SikhiToTheMax.app",
                    f"{os.path.expanduser('~')}/Applications/SikhiToTheMax.app"
                ]
                
                for path in app_paths:
                    if os.path.exists(path):
                        logger.info(f"Found STTM at {path}. Opening...")
                        subprocess.run(['open', path])
                        time.sleep(3)
                        break
        
        # Now activate it (bring to front) if it's running
        activate_script = '''
        tell application "SikhiToTheMax" to activate
        tell application "System Events"
            tell process "SikhiToTheMax"
                set frontmost to true
            end tell
        end tell
        '''
        
        subprocess.run(['osascript', '-e', activate_script], capture_output=True)
        
        # Verify that STTM is now the frontmost app
        verify_script = '''
        tell application "System Events"
            set frontApp to name of first process whose frontmost is true
            return frontApp
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', verify_script], capture_output=True, text=True)
        front_app = result.stdout.strip()
        
        if "SikhiToTheMax" in front_app:
            logger.info("STTM application activated successfully")
        else:
            logger.warning(f"STTM activation verification failed. Front app is: {front_app}")
            
    except Exception as e:
        logger.error(f"Error activating STTM application: {e}")
        raise

def click_at_coordinates(x, y):
    """
    Click at the specified coordinates using both PyAutoGUI and AppleScript for reliability
    
    Args:
        x (int): X coordinate
        y (int): Y coordinate
    """
    try:
        logger.info(f"Clicking at coordinates ({x}, {y})")
        
        # Make sure STTM is active before clicking
        activate_script = '''
        tell application "SikhiToTheMax" to activate
        delay 1
        tell application "System Events"
            tell process "SikhiToTheMax"
                set frontmost to true
            end tell
        end tell
        '''
        subprocess.run(['osascript', '-e', activate_script], capture_output=True)

        try:
            # This method directly clicks at screen coordinates through System Events
            apple_script = f'''
            tell application "System Events"
                click at {{{x}, {y}}}
            end tell
            '''
            result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
            logger.info(f"System Events click completed: {result.stdout if result.stdout else 'No output'}")
            
            if result.stderr:
                logger.warning(f"System Events click warning: {result.stderr}")
        except Exception as e:
            logger.warning(f"System Events click failed: {e}")
            
        # Method 3: Attempt click directly on STTM process if it can be identified
        try:
            # This method tries to click within the process context
            apple_script = f'''
            tell application "System Events"
                tell process "SikhiToTheMax"
                    click at {{{x}, {y}}}
                end tell
            end tell
            '''
            result = subprocess.run(['osascript', '-e', apple_script], capture_output=True, text=True)
            logger.info(f"Process-specific click completed: {result.stdout if result.stdout else 'No output'}")
            
            if result.stderr:
                logger.warning(f"Process-specific click warning: {result.stderr}")
        except Exception as e:
            logger.warning(f"Process-specific click failed: {e}")
        
    except Exception as e:
        logger.error(f"Error clicking at coordinates ({x}, {y}): {e}")
        raise

def open_sync_panel():
    """
    Click on the phone icon at the fixed coordinates to open the sync panel
    Using cliclick for more reliable clicking on macOS
    """
    try:
        logger.info("Opening sync panel")
        
        # First make sure STTM is active
        activate_sttm()
        
        # Use cliclick for more reliable clicking on macOS
        try:
            subprocess.run(['cliclick', f'c:{PHONE_ICON_X},{PHONE_ICON_Y}'], check=True)
        except subprocess.CalledProcessError:
            logger.warning("cliclick failed, falling back to AppleScript")
            apple_script = f'''
            tell application "System Events"
                tell process "SikhiToTheMax"
                    click at {{{PHONE_ICON_X}, {PHONE_ICON_Y}}}
                end tell
            end tell
            '''
            subprocess.run(['osascript', '-e', apple_script], capture_output=True)
        except FileNotFoundError:
            logger.warning("cliclick not found, falling back to AppleScript")
            apple_script = f'''
            tell application "System Events"
                tell process "SikhiToTheMax"
                    click at {{{PHONE_ICON_X}, {PHONE_ICON_Y}}}
                end tell
            end tell
            '''
            subprocess.run(['osascript', '-e', apple_script], capture_output=True)
        
        time.sleep(1)  # Wait for panel to open
    except Exception as e:
        logger.error(f"Error opening sync panel: {e}")
        raise

def close_sync_panel():
    """
    Click on the phone icon again to close the sync panel.
    Using cliclick for more reliable clicking on macOS
    """
    try:
        
        # Use cliclick for more reliable clicking on macOS
        try:
            subprocess.run(['cliclick', f'c:{PHONE_ICON_X},{PHONE_ICON_Y}'], check=True)
        except subprocess.CalledProcessError:
            logger.warning("cliclick failed, falling back to AppleScript")
            apple_script = f'''
            tell application "System Events"
                tell process "SikhiToTheMax"
                    click at {{{PHONE_ICON_X}, {PHONE_ICON_Y}}}
                end tell
            end tell
            '''
            subprocess.run(['osascript', '-e', apple_script], capture_output=True)
        except FileNotFoundError:
            logger.warning("cliclick not found, falling back to AppleScript")
            apple_script = f'''
            tell application "System Events"
                tell process "SikhiToTheMax"
                    click at {{{PHONE_ICON_X}, {PHONE_ICON_Y}}}
                end tell
            end tell
            '''
            subprocess.run(['osascript', '-e', apple_script], capture_output=True)
        
    except Exception as e:
        logger.error(f"Error closing sync panel: {e}")
        raise

def capture_screen():
    """
    Capture a screenshot of the screen (only the left third)
    
    Returns:
        PIL.Image: Screenshot image of the left third of the screen
    """
    try:
        full_screenshot = ImageGrab.grab()
        
        # Crop just the left third of the screen
        width, height = full_screenshot.size
        left_third = full_screenshot.crop((0, 0, width // 3, height))
        
        return left_third
    except Exception as e:
        logger.error(f"Error capturing screen: {e}")
        raise

def extract_sync_code_pin():
    """
    Extract the sync code and PIN from the sync panel using OCR with pytesseract
    
    Returns:
        tuple: (sync_code, pin) or (None, None) if extraction fails
    """
    try:
        
        # Capture a screenshot
        screenshot = capture_screen()
        
        # Save the screenshot to the current directory
        try:
            # Convert to RGB mode (removes alpha channel) before saving as JPEG
            if screenshot.mode == 'RGBA':
                screenshot = screenshot.convert('RGB')
                
            screenshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_panel.jpg")
            screenshot.save(screenshot_path)
        except Exception as img_error:
            logger.warning(f"Could not save screenshot: {img_error}")
            return None, None
        
        try:
            width, height = screenshot.size
            
            panel_top = height // 3
            panel_bottom = 2 * height // 3
            
            # Save this cropped section as a separate file for debugging
            sync_panel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_panel.jpg")
            screenshot.save(sync_panel_path)
            logger.info(f"Saved sync panel crop to {sync_panel_path}")

            
            # Perform OCR with optimized parameters
            # --oem 3: Use LSTM neural network
            # --psm 6: Assume a single uniform block of text
            # -l eng: Force English language
            ocr_text = pytesseract.image_to_string(
                screenshot,
                config='--oem 3 -l eng'
            )

            sync_code_pattern = r'[A-Z]{3}-[A-Z]{3}'  # Standard format with hyphen

            sync_code = None
            sync_code_match = re.search(sync_code_pattern, ocr_text)
            if sync_code_match:
                # If the pattern has a capture group, use that, otherwise use the whole match
                if len(sync_code_match.groups()) > 0:
                    sync_code = sync_code_match.group(1)
                else:
                    sync_code = sync_code_match.group(0)
                    
            if len(sync_code) == 6 and '-' not in sync_code:
                sync_code = f"{sync_code[:3]}-{sync_code[3:]}"

            if not sync_code:
                logger.warning("Could not extract sync code with any regex pattern")
            
            # For PIN (typically 4 digits)
            pin_pattern = r'Pin:?\s*(\d{4})'  # Standard format with "Pin:" prefix

            pin = None
            pin_match = re.search(pin_pattern, ocr_text)
            if pin_match:
                pin = pin_match.group(1)
                    # Ensure it's exactly 4 digits
                if len(pin) != 4:
                    logger.warning(f"PIN length is not 4: {pin}")
                    # If longer or shorter, try to extract 4 digits
                    if len(pin) > 4:
                        pin = pin[:4]
                    else:
                        pin = pin.zfill(4)  # Pad with zeros if shorter
            
            # Process the extracted values
            if sync_code and pin:
                logger.info(f"Successfully extracted sync code: {sync_code}, PIN: {pin}")
                return sync_code, pin
            else:
                logger.error(f"Failed to extract sync code and PIN. OCR result: {ocr_text}")
                raise ValueError("Could not extract sync code and PIN from OCR text")
                
        except Exception as ocr_error:
            logger.error(f"OCR processing failed: {ocr_error}")
            raise
    
    except Exception as e:
        logger.error(f"Error extracting sync code and PIN: {e}")
        return None, None

if __name__ == "__main__":
    # Simple test for debugging
    activate_sttm()
    open_sync_panel()
    time.sleep(0.5)
    close_sync_panel()
    print("Test completed successfully")
