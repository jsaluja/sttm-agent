import requests
import socketio
import time
import re
from typing import Dict, Any, Optional, Union

# Disable all Socket.IO related logging to suppress verbose output
import logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger('websocket').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

class STTMSyncController:
    """
    Controller for SikhiToTheMax (STTM) sync service
    Uses a simplified approach based on the working chat_working.py implementation
    """
    def __init__(self, sync_api_url: str):
        """
        Initialize the STTM Sync Controller
        
        Args:
            sync_api_url: Base URL for the sync API (e.g., "https://api.sikhitothemax.org/")
        """
        self.sync_api_url = sync_api_url.rstrip('/') + '/'
        self.socket = None
        self.connected = False
        self.namespace_string = ""
        self.socket_namespace = None
        self.controller_pin = 0
        self.auth_completed = False
        
        # Standard headers based on browser behavior
        self.headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.sikhitothemax.org",
            "Sec-Ch-Ua": "\"Not:A=Brand\";v=\"99\", \"Google Chrome\";v=\"139\", \"Chromium\";v=\"139\"",
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": "\"macOS\"",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
            "Referer": "https://www.sikhitothemax.org/"
        }

    def connect(self, code: str, pin: Union[str, int]) -> Dict[str, Any]:
        """
        Connect to the sync service using code and pin
        
        Args:
            code: Unique sync code (e.g., "ABC-XYZ")
            pin: Admin PIN for authentication
            
        Returns:
            Dictionary with connection status and any error information
        """
        try:
            # Reset state
            self.connected = False
            self.auth_completed = False
            
            # Store the pin (try converting to int if possible)
            try:
                self.controller_pin = int(pin)
            except ValueError:
                self.controller_pin = pin
            
            # STEP 1: Pre-fetch app.js to simulate browser behavior
            requests.get("https://www.sikhitothemax.org/app.js", headers=self.headers)
            
            # STEP 2: Join the sync session using the code
            join_url = f"{self.sync_api_url}sync/join/{code}"
            join_resp = requests.get(join_url, headers=self.headers, timeout=10)
            
            if join_resp.status_code != 200:
                return {
                    'success': False,
                    'error': f"Failed to join session: {join_resp.status_code} {join_resp.text}",
                    'error_type': 'network_error'
                }
            
            session_data = join_resp.json()
            
            # Check for API-level errors
            if session_data.get('error') or session_data.get('data') is None:
                return {
                    'success': False,
                    'error': session_data.get('error', 'No data received'),
                    'error_type': 'code_error'
                }
            
            # Extract namespace for socket connection
            self.namespace_string = session_data['data']['namespaceString']
            namespace = "/" + self.namespace_string
            self.socket_namespace = namespace
            
            # STEP 3: Set up Socket.IO client - disable all logging
            self.socket = socketio.Client(logger=False, engineio_logger=False)
            
            # Set up event handlers
            @self.socket.event(namespace=namespace)
            def connect():
                # Send authentication request on connect
                self._send_auth_request()
            
            @self.socket.event
            def disconnect():
                self.connected = False
                print("❌ Disconnected from Socket.IO server")
            
            @self.socket.on("data", namespace=namespace)
            def on_data(data):
                # Handle authentication response
                if isinstance(data, dict) and data.get("type") == "response-control":
                    success_pin = data.get("success")
                    if success_pin == self.controller_pin:
                        print(f"✅ Connected to Server with pin {success_pin}")
                        self.connected = True
                        self.auth_completed = True
                    else:
                        print(f"⚠️ Pin mismatch! Server responded with {success_pin}")
            
            @self.socket.event
            def connect_error(data):
                print("⚠️ Connection error:", data)
            
            # STEP 4: Connect to Socket.IO server
            print(f"Connecting to STTM with sync code {code}...")
            self.socket.connect(
                self.sync_api_url,
                socketio_path="socket.io",
                headers=self.headers
            )
            
            # Wait for authentication to complete
            print("Authenticating with server...")
            wait_count = 0
            while not self.auth_completed and wait_count < 10:
                time.sleep(0.5)
                wait_count += 1
            
            # Return success
            return {
                'success': True,
                'namespace': self.namespace_string,
                'immediate': True
            }
            
        except requests.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': 'network_error'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': 'general_error'
            }
    
    def _send_auth_request(self):
        """Send authentication request to the server"""
        auth_payload = {
            "host": "sttm-web",
            "type": "request-control",
            "pin": str(self.controller_pin) if isinstance(self.controller_pin, int) else self.controller_pin
        }
        self.socket.emit("data", auth_payload, namespace=self.socket_namespace)
    
    def send_shabad(self, shabad_id: int, verse_id: Optional[int] = None) -> bool:
        """
        Send a shabad to the STTM server
        
        Args:
            shabad_id: The ID of the shabad
            verse_id: The ID of the verse (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected or not self.socket:
            print("⚠️ Not connected to STTM server")
            return False
        
        try:
            # Convert IDs to integers if they're strings
            shabad_id = int(shabad_id)
            if verse_id is not None:
                verse_id = int(verse_id)
            
            # Build shabad payload
            shabad_payload = {
                "host": "sttm-web",
                "type": "shabad",
                "pin": self.controller_pin,
                "shabadId": shabad_id
            }
            
            # Add verse ID if provided
            if verse_id is not None:
                shabad_payload["verseId"] = verse_id
                
            # Send the payload
            self.socket.emit("data", shabad_payload, namespace=self.socket_namespace)
            print(f"📤 Event sent for Shabad {shabad_id}" + (f" Verse {verse_id}" if verse_id else ""))
            return True
            
        except Exception as e:
            print(f"❌ Error sending shabad: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the sync service"""
        if self.socket and self.socket.connected:
            self.socket.disconnect()
        self.connected = False
        self.namespace_string = ""
        self.socket_namespace = None
        self.controller_pin = 0
        self.auth_completed = False

def send_shabad(controller, shabad_id, verse_id=None):
    """Send a shabad to the connected desktop app"""
    try:
        # Convert IDs to integers
        shabad_id = int(shabad_id)
        if verse_id is not None:
            verse_id = int(verse_id)
            print(f"✅ Sending shabad ID {shabad_id} with verse ID {verse_id}")
        else:
            print(f"✅ Sending shabad ID {shabad_id}")
        
        # Send the shabad
        result = controller.send_shabad(shabad_id, verse_id)
        return result
    except ValueError:
        print("❌ Error: Shabad ID and verse ID must be integers")
        return False
    except Exception as e:
        print(f"❌ Error sending shabad: {str(e)}")
        return False

def main():
    """Main function for direct usage"""
    # Use the production STTM sync API
    controller = STTMSyncController("https://api.sikhitothemax.org/")
    
    # Get sync code and PIN from user
    sync_code = input("Enter sync code from desktop (e.g., ABC-XYZ): ").strip().upper()
    admin_pin = input("Enter admin PIN: ").strip()
    
    print("Connecting to STTM sync service...")
    
    # Connect to the service
    result = controller.connect(sync_code, admin_pin)
    
    if result.get('success'):
        print(f"✅ Connected successfully!")
        print(f"📡 Namespace: {result.get('namespace', 'Unknown')}")
        
        # Command interface
        print("\n📜 STTM Command Interface")
        print("Commands:")
        print("  <shabad_id> <verse_id>   - Control shabad and verse")
        print("  exit/quit                - Disconnect and exit")
        print("\nWaiting for commands...")
        
        try:
            while True:
                cmd_input = input("\nEnter command: ").strip()
                
                # Exit commands
                if cmd_input.lower() in ['exit', 'quit']:
                    break
                    
                # Process shabad commands
                parts = cmd_input.split()
                
                if len(parts) == 2:
                    # Shabad ID and verse ID
                    send_shabad(controller, parts[0], parts[1])
                else:
                    print("❌ Invalid command format. Use: <shabad_id> <verse_id>")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            controller.disconnect()
            print("🔌 Disconnected from sync service")
    else:
        error_type = result.get('error_type', 'unknown_error')
        print(f"❌ Connection failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
