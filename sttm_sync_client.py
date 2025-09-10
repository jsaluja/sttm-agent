import sys
import os

# Ensure sync.py is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from sync.py
from sync import STTMSyncController, send_shabad

class STTMSyncClient:
    """
    Wrapper for STTMSyncController with simplified interface
    for integration with the agent
    """
    
    def __init__(self):
        self.controller = None
        self.connected = False
        
    def connect_with_code_pin(self, sync_code, pin):
        """
        Connect to STTM using the provided sync code and PIN
        
        Args:
            sync_code: The sync code from the STTM app (e.g., "ABC-XYZ")
            pin: The admin PIN (e.g., "1234")
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        if not sync_code or not pin:
            print("⚠️ Missing sync code or PIN")
            return False
            
        # Initialize controller if not already done
        if self.controller is None:
            self.controller = STTMSyncController("https://api.sikhitothemax.org/")
        
        # Connect to the sync service
        print(f"Connecting to STTM with code: {sync_code}, PIN: {pin}")
        result = self.controller.connect(sync_code, pin)
        
        # Check if connection was successful
        if result.get('success'):
            print(f"✅ Connected successfully to STTM!")
            self.connected = True
            return True
        else:
            error_type = result.get('error_type', 'unknown_error')
            print(f"❌ Connection failed: {result.get('error', 'Unknown error')}")
            self.connected = False
            return False
    
    def send_verse(self, shabad_id, verse_id):
        """
        Send a verse to STTM
        
        Args:
            shabad_id: The ID of the shabad
            verse_id: The ID of the verse within the shabad
            
        Returns:
            bool: True if sending was successful, False otherwise
        """
        if not self.connected or self.controller is None:
            print("⚠️ Not connected to STTM")
            return False
            
        try:
            # Convert IDs to integers if they're not already
            if isinstance(shabad_id, str):
                shabad_id = int(shabad_id)
            if isinstance(verse_id, str):
                verse_id = int(verse_id)
                
            # Send the verse
            print(f"Sending verse: shabad_id={shabad_id}, verse_id={verse_id}")
            success = send_shabad(self.controller, shabad_id, verse_id)
            return success
        except Exception as e:
            print(f"❌ Error sending verse: {str(e)}")
            return False
            
    def disconnect(self):
        """
        Disconnect from STTM
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        if self.controller is None:
            return True
            
        try:
            self.controller.disconnect()
            self.connected = False
            print("🔌 Disconnected from STTM")
            return True
        except Exception as e:
            print(f"❌ Error disconnecting: {str(e)}")
            return False
