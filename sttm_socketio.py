import socketio
import requests
import json
import time
import logging
import sys

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Create logger for this module
logger = logging.getLogger(__name__)

class STTMSyncSocketIO:
    def __init__(self):
        """Initialize STTM Sync Socket.IO client"""
        self.base_url = "https://api.sikhitothemax.org"
        self.namespace = None
        self.pin = None
        self.socket = None
        self.connected = False
        self.socket_namespace = None
        self.auth_completed = False
        self.auth_success = False

    def connect(self, sync_code, pin):
        """
        Connect to STTM sync using the provided code and pin
        
        Args:
            sync_code: The sync code from desktop app (e.g., "ABC-XYZ")
            pin: The PIN for controller access
        
        Returns:
            Dict with connection status
        """
        try:
            logger.info(f"Starting connection with code: {sync_code}, pin: {pin}")
            self.pin = pin
            
            # Step 1: Request namespace string
            logger.info("Step 1: Requesting namespace string")
            join_url = f"{self.base_url}/sync/join/{sync_code}"
            
            response = requests.get(join_url)
            logger.info(f"Response: {response.text}")
            response.raise_for_status()
            
            result = response.json()
            
            # Check for errors or missing data
            if result.get('error') or not result.get('data'):
                error_message = result.get('error', 'No data received')
                logger.error(f"Failed to join sync: {error_message}")
                return {'success': False, 'error': error_message}
            
            # Extract namespace
            self.namespace = result['data']['namespaceString']
            logger.info(f"Extracted namespace: {self.namespace}")
            
            # Step 2: Connect to socket.io
            logger.info("Step 2: Connecting to socket.io")
            logger.info(f"Using base URL: {self.base_url}")
            namespace = f"/{self.namespace}"
            logger.info(f"Using namespace: {namespace}")
            
            # Create a socket.io client with proper configuration
            self.socket = socketio.Client(logger=False, reconnection=True, reconnection_attempts=3)
            
            # Set up socket event handlers
            self._setup_socket_handlers()
            
            # Attempt to connect with namespace parameter
            try:
                logger.info(f"Connecting to socket.io with namespace parameter...")
                self.socket.connect(
                    f"{self.base_url}",
                    transports=["polling"],
                    namespaces=[namespace],
                    wait_timeout=10
                )
                self.socket_namespace = namespace
                self.connected = True
                logger.info("Socket connected successfully!")
            except socketio.exceptions.ConnectionError as e:
                logger.error(f"Socket.IO connection failed: {str(e)}")
                logger.info("Trying alternative URLs:")
                
                # Try connecting with namespace in URL
                try:
                    url_with_namespace = f"{self.base_url}{namespace}"
                    logger.info(f"Trying URL with namespace: {url_with_namespace}")
                    self.socket.connect(url_with_namespace, transports=["polling"])
                    self.connected = True
                    logger.info("Socket connected successfully!")
                except Exception as e:
                    logger.error(f"Connection failed: {str(e)}")
                    
                    # Try connecting with engineio path
                    try:
                        logger.info(f"Trying URL with engineio path: {self.base_url}")
                        self.socket.connect(
                            f"{self.base_url}", 
                            transports=["polling"], 
                            wait=True,
                            engineio_path="socket.io"
                        )
                        self.connected = True
                        logger.info("Socket connected successfully!")
                    except Exception as e:
                        logger.error(f"Connection failed: {str(e)}")
                        logger.error("All connection attempts failed.")
                        return {'success': False, 'error': 'Could not establish socket.io connection'}
            
            # If we get here, we've connected successfully
            # Now authenticate with the PIN
            if self.connected:
                logger.info("Step 3: Authenticating with PIN")
                self._send_control_request()
                
                # Wait for authentication response
                timeout = time.time() + 5  # 5 seconds timeout
                while not self.auth_completed and time.time() < timeout:
                    logger.info("Waiting for auth response...")
                    time.sleep(0.5)
                
                if self.auth_success:
                    logger.info("Authentication successful!")
                    return {'success': True, 'namespace': self.namespace}
                else:
                    logger.error("Authentication failed or timed out")
                    return {'success': False, 'error': 'Authentication failed'}
            else:
                return {'success': False, 'error': 'Socket connection failed'}
                
        except requests.RequestException as e:
            logger.error(f"Network error: {str(e)}")
            return {'success': False, 'error': f"Network error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error connecting: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _setup_socket_handlers(self):
        """Set up socket.io event handlers"""
        
        @self.socket.event
        def connect():
            logger.info("Socket connected")
            self.connected = True
        
        @self.socket.event
        def disconnect():
            logger.info("Socket disconnected")
            self.connected = False
        
        @self.socket.event
        def data(data):
            """Handle incoming data from socket"""
            logger.info(f"Received data: {json.dumps(data)}")
            
            if data.get('type') == 'response-control':
                logger.info("Received control response")
                self.auth_completed = True
                
                if data.get('success'):
                    logger.info("Authentication successful!")
                    self.auth_success = True
                else:
                    logger.error(f"Authentication failed: {data}")
                    self.auth_success = False
            
            elif data.get('type') in ['shabad', 'ceremony', 'bani', 'settings']:
                logger.info(f"Received {data.get('type')} data")
                # If we're receiving data from desktop, we're likely authenticated
                if not self.auth_completed and data.get('host') == 'sttm-desktop':
                    logger.info("Assuming authentication was successful based on desktop data")
                    self.auth_completed = True
                    self.auth_success = True
    
    def _send_control_request(self):
        """Send control request with PIN for authentication"""
        if not self.connected or not self.socket:
            logger.error("Cannot send control request - not connected")
            return False
        
        try:
            # Create control message
            control_message = {
                'host': 'sttm-web',
                'type': 'control',
                'pin': self.pin
            }
            
            logger.info(f"Sending control message: {json.dumps(control_message)}")
            
            if self.socket_namespace:
                self.socket.emit('data', control_message, namespace=self.socket_namespace)
            else:
                self.socket.emit('data', control_message)
            
            logger.info("Control message sent")
            return True
        except Exception as e:
            logger.error(f"Error sending control message: {str(e)}")
            return False
    
    def send_shabad(self, shabad_id, verse_id=None):
        """
        Send a shabad to the connected desktop app
        """
        if not self.connected or not self.socket:
            logger.error("Not connected")
            return {'success': False, 'error': 'Not connected'}
        
        try:
            # Convert IDs to integers
            shabad_id = int(shabad_id)
            verse_id = int(verse_id) if verse_id is not None else None
            
            # Create shabad message
            message = {
                'host': 'sttm-web',
                'type': 'shabad',
                'pin': self.pin,
                'shabadId': shabad_id
            }
            
            if verse_id is not None:
                message['verseId'] = verse_id
            
            logger.info(f"Sending shabad: {json.dumps(message)}")
            
            if self.socket_namespace:
                self.socket.emit('data', message, namespace=self.socket_namespace)
            else:
                self.socket.emit('data', message)
            
            logger.info("Shabad message sent")
            return {'success': True}
        except ValueError:
            logger.error("Shabad ID and verse ID must be integers")
            return {'success': False, 'error': 'Shabad ID and verse ID must be integers'}
        except Exception as e:
            logger.error(f"Error sending shabad: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def disconnect(self):
        """Disconnect from socket.io server"""
        if self.socket and self.connected:
            self.socket.disconnect()
        self.connected = False
        self.namespace = None
        self.socket_namespace = None
        self.auth_completed = False
        self.auth_success = False

def main():
    """Main entry point"""
    logger.info("STTM Sync Socket.IO Client")
    
    # Get sync code and PIN
    sync_code = input("Enter sync code (e.g., ABC-XYZ): ").strip().upper()
    pin = input("Enter PIN: ").strip()
    
    client = STTMSyncSocketIO()
    
    # Connect to sync server
    logger.info("Connecting to STTM sync...")
    result = client.connect(sync_code, pin)
    
    if result['success']:
        logger.info(f"Connected successfully to namespace: {result['namespace']}")
        
        print("\n--- STTM Control Interface ---")
        print("Commands:")
        print("  <shabad_id> <verse_id>  - Control shabad (verse_id optional)")
        print("  exit/quit               - Disconnect and exit")
        
        try:
            while True:
                command = input("\nEnter command: ").strip()
                
                if command.lower() in ['exit', 'quit']:
                    break
                    
                parts = command.split()
                if not parts:
                    continue
                    
                try:
                    if len(parts) == 1:
                        # Just shabad ID
                        shabad_id = parts[0]
                        result = client.send_shabad(shabad_id)
                    elif len(parts) >= 2:
                        # Shabad ID and verse ID
                        shabad_id = parts[0]
                        verse_id = parts[1]
                        result = client.send_shabad(shabad_id, verse_id)
                    
                    if result['success']:
                        print("Command sent successfully")
                    else:
                        print(f"Command failed: {result.get('error', 'Unknown error')}")
                        
                except ValueError:
                    print("Invalid input. Shabad ID and verse ID must be integers.")
                except Exception as e:
                    print(f"Error processing command: {str(e)}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            client.disconnect()
            logger.info("Disconnected")
    else:
        logger.error(f"Connection failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
