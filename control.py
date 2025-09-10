import subprocess
import time
from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
import subprocess
import os

APP_BUNDLE_ID = "org.khalisfoundation.sttm"     # e.g. 'com.example.myapp'
APP_NAME = "SikhiToTheMax"           # e.g. 'MyApp'
TOOLTIP = "Bani Controller"

def open_app(bundle_id, app_name):
    """
    Open the app by bundle id if not running.
    """
    try:
        # Find the running applications with the given bundle id
        running_apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)
        app_running = len(running_apps) > 0
        
        if not app_running:
            # Not running, open app
            subprocess.Popen(['open', '-b', bundle_id])
            # Wait for app to start (max 15s)
            for _ in range(30):
                time.sleep(0.5)
                running_apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)
                if len(running_apps) > 0:
                    break
                    
        if len(running_apps) > 0:
            # App is running, return True
            return True
        return False
    except Exception as e:
        print(f"Error launching app: {e}")
        return False

def activate_app(bundle_id):
    """
    Brings app window to front.
    """
    try:
        running_apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)
        if len(running_apps) > 0:
            app = running_apps[0]
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            time.sleep(1)  # Let macOS UI settle
            return True
        return False
    except Exception as e:
        print(f"Error activating app: {e}")
        return False

def list_ui_elements(app_name):
    """
    Use AppleScript to list ALL UI elements and their properties for debugging.
    This function goes deeper into the UI hierarchy.
    """
    script_path = '/tmp/list_elements.scpt'
    with open(script_path, 'w') as f:
        f.write(f'''
tell application "System Events"
    tell process "{app_name}"
        set elementList to ""
        try
            set windowList to windows
            repeat with w in windowList
                set elementList to elementList & "Window: " & name of w & return
                
                -- Get all UI elements recursively
                on get_elements_info(el, indent_level)
                    set result_text to ""
                    set indent_str to ""
                    repeat indent_level times
                        set indent_str to indent_str & "  "
                    end repeat
                    
                    -- Get element info
                    try
                        set el_class to class of el as text
                        set result_text to result_text & indent_str & el_class & ": "
                        
                        try
                            set el_name to name of el
                            if el_name is not missing value then
                                set result_text to result_text & el_name
                            else
                                set result_text to result_text & "[unnamed]"
                            end if
                        end try
                        
                        try
                            set el_role to role of el
                            if el_role is not missing value and el_role is not "" then
                                set result_text to result_text & " (Role: " & el_role & ")"
                            end if
                        end try
                        
                        try
                            set el_help to help of el
                            if el_help is not missing value and el_help is not "" then
                                set result_text to result_text & " (Help: " & el_help & ")"
                            end if
                        end try
                        
                        try
                            set el_desc to description of el
                            if el_desc is not missing value and el_desc is not "" then
                                set result_text to result_text & " (Desc: " & el_desc & ")"
                            end if
                        end try
                        
                        try
                            set el_title to title of el
                            if el_title is not missing value and el_title is not "" then
                                set result_text to result_text & " (Title: " & el_title & ")"
                            end if
                        end try
                        
                        try
                            set el_value to value of el
                            if el_value is not missing value and el_value is not "" then
                                set result_text to result_text & " (Value: " & el_value & ")"
                            end if
                        end try
                        
                        set result_text to result_text & return
                        
                        -- Process children
                        try
                            set child_elements to UI elements of el
                            repeat with child in child_elements
                                set result_text to result_text & my get_elements_info(child, indent_level + 1)
                            end repeat
                        end try
                    end try
                    
                    return result_text
                end get_elements_info
                
                -- Get elements info starting from window
                set elementList to elementList & get_elements_info(w, 1)
            end repeat
        end try
        return elementList
    end tell
end tell
''')
    
    try:
        result = subprocess.run(['osascript', script_path], capture_output=True, text=True)
        print(f"UI Elements:\n{result.stdout.strip()}")
        os.remove(script_path)
    except Exception as e:
        print(f"Error executing AppleScript: {e}")
        if os.path.exists(script_path):
            os.remove(script_path)

def find_and_click_by_tooltip(app_name, tooltip):
    """
    Use AppleScript to search for and click a UI element with the given tooltip.
    """
    # First list all elements for debugging
    list_ui_elements(app_name)
    
    # Create a temporary AppleScript file for a deep recursive search
    script_path = '/tmp/click_button.scpt'
    with open(script_path, 'w') as f:
        f.write(f'''
tell application "System Events"
    tell process "{app_name}"
        -- Recursive function to search for elements
        on find_element_by_property(el, prop_name, prop_value)
            -- Check if this element has the property we're looking for
            try
                if el has prop_name then
                    set prop_result to el's prop_name
                    if prop_result contains prop_value then
                        click el
                        return "Clicked element with " & prop_name & ": " & prop_result
                    end if
                end if
            end try
            
            -- Check children recursively
            try
                set child_elements to UI elements of el
                repeat with child in child_elements
                    set search_result to my find_element_by_property(child, prop_name, prop_value)
                    if search_result is not false then
                        return search_result
                    end if
                end repeat
            end try
            
            return false
        end find_element_by_property
        
        -- Try to find by different properties
        try
            set allWindows to windows
            repeat with w in allWindows
                -- Try help text
                set result to my find_element_by_property(w, "help", "{tooltip}")
                if result is not false then
                    return result
                end if
                
                -- Try description
                set result to my find_element_by_property(w, "description", "{tooltip}")
                if result is not false then
                    return result
                end if
                
                -- Try title
                set result to my find_element_by_property(w, "title", "{tooltip}")
                if result is not false then
                    return result
                end if
                
                -- Try name
                set result to my find_element_by_property(w, "name", "{tooltip}")
                if result is not false then
                    return result
                end if
            end repeat
        end try
        
        -- Not found
        return "No element found with '" & "{tooltip}" & "' property"
    end tell
end tell
''')
    
    # Execute the AppleScript
    try:
        result = subprocess.run(['osascript', script_path], capture_output=True, text=True)
        print(f"AppleScript output: {result.stdout.strip()}")
        # Clean up
        os.remove(script_path)
        return "Button clicked" in result.stdout
    except Exception as e:
        print(f"Error executing AppleScript: {e}")
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
        return False

def try_direct_click(app_name, target_name):
    """
    Try to directly click on a menu item or button by name or title.
    """
    script_path = '/tmp/direct_click.scpt'
    with open(script_path, 'w') as f:
        f.write(f'''
tell application "System Events"
    tell process "{app_name}"
        try
            -- Try to find menu items with the name
            set menuFound to false
            tell menu bar 1
                try
                    set allMenus to menu bar items
                    repeat with m in allMenus
                        try
                            click menu item "{target_name}" of menu of m
                            set menuFound to true
                            exit repeat
                        end try
                    end repeat
                end try
            end tell
            
            if menuFound then
                return "Clicked menu item: {target_name}"
            end if
            
            -- Try to find a button with the name
            set windowList to windows
            repeat with w in windowList
                try
                    click button "{target_name}" of w
                    return "Clicked button: {target_name}"
                end try
            end repeat
            
            return "Could not find '{target_name}'"
        end try
    end tell
end tell
''')
    
    try:
        result = subprocess.run(['osascript', script_path], capture_output=True, text=True)
        print(f"Direct click result: {result.stdout.strip()}")
        os.remove(script_path)
        return "Clicked" in result.stdout
    except Exception as e:
        print(f"Error with direct click: {e}")
        if os.path.exists(script_path):
            os.remove(script_path)
        return False

def main():
    # Launch the app if not running
    if not open_app(APP_BUNDLE_ID, APP_NAME):
        print("App could not be launched or located.")
        return
    
    # Activate the app
    if not activate_app(APP_BUNDLE_ID):
        print("App could not be activated.")
        return
    
    # Give the app time to fully activate
    time.sleep(1)
    
    # First try a direct click on common UI elements
    print(f"Trying to directly click on '{TOOLTIP}'...")
    if try_direct_click(APP_NAME, "Bani Controller"):
        print("Clicked on target directly.")
        return
        
    # Try to find a close match - maybe it's a menu item called "Controller"
    print("Trying 'Controller'...")
    if try_direct_click(APP_NAME, "Controller"):
        print("Clicked on 'Controller'.")
        return
    
    # As a last resort, try to find by tooltip
    print("Trying to find by tooltip...")
    if find_and_click_by_tooltip(APP_NAME, TOOLTIP):
        print("Clicked on element with tooltip:", TOOLTIP)
    else:
        print("Could not find any matching elements. You may need to manually inspect the app's UI.")

if __name__ == "__main__":
    main()
