import requests
import socketio
import time

# === CONFIGURATION ===
SESSION_ID = "XUF-SXJ" 
PIN_CODE = 9362

HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.sikhitothemax.org",
    "Sec-Ch-Ua": "\"Not:A=Brand\";v=\"99\", \"Google Chrome\";v=\"139\", \"Chromium\";v=\"139\"",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "\"macOS\"",
    "User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "Referer": "https://www.sikhitothemax.org/"
}

SOCKETIO_PATH = "socket.io"
SOCKETIO_URL = "https://api.sikhitothemax.org"

# === STEP 1: JOIN SESSION ===
requests.get("https://www.sikhitothemax.org/app.js", headers=HEADERS)
join_resp = requests.get(f"https://api.sikhitothemax.org/sync/join/{SESSION_ID}",
                         headers=HEADERS)
if join_resp.status_code != 200:
    raise RuntimeError(f"Failed to join session: {join_resp.status_code} {join_resp.text}")

session_data = join_resp.json()
namespace = "/" + session_data["data"]["namespaceString"]

# === STEP 2: CONNECT VIA SOCKET.IO ===
sio = socketio.Client(logger=False, engineio_logger=False)

@sio.event(namespace=namespace)
def connect():
    request_payload = {
        "host": "sttm-web",
        "type": "request-control",
        "pin": str(PIN_CODE)
    }
    sio.emit("data", request_payload, namespace=namespace)

@sio.event
def disconnect():
    print("❌ Disconnected from Socket.IO server")

# Note: on_data handler is now defined before the main function

@sio.event
def connect_error(data):
    print("⚠️ Connection error:", data)

# Flag to track authentication status
auth_completed = False

@sio.on("data", namespace=namespace)
def on_data(data):
    global auth_completed
    # Validate response-control
    if isinstance(data, dict) and data.get("type") == "response-control":
        success_pin = data.get("success")
        if success_pin == PIN_CODE:
            print(f"✅ Connected to Server with pin {success_pin}")
            auth_completed = True
        else:
            print(f"⚠️ Pin mismatch! Server responded with {success_pin}")

# Function to send a shabad to the STTM server
def send_shabad(shabad_id: str, verse_id: str) -> bool:
    """
    Send a shabad to the STTM server
    
    Args:
        shabad_id: The ID of the shabad
        verse_id: The ID of the verse
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        shabad_id = int(shabad_id)
        verse_id = int(verse_id)
    except ValueError:
        print("⚠️ Invalid input: Shabad ID and Verse ID must be integers.")
        return False

    # Build shabad payload
    shabad_payload = {
        "host": "sttm-web",
        "type": "shabad",
        "pin": int(PIN_CODE),
        "shabadId": shabad_id,
        "verseId": verse_id,
    }
    
    # Send the payload
    try:
        sio.emit("data", shabad_payload, namespace=namespace)
        print("📤 Event sent for Shabad {} Verse {}".format(shabad_id, verse_id))
        return True
    except Exception as e:
        print(f"❌ Error sending shabad: {e}")
        return False

# === STEP 3: MAIN SCRIPT ===
def main():
    try:
        sio.connect(
            SOCKETIO_URL,
            socketio_path=SOCKETIO_PATH,
            headers=HEADERS
        )
        
        # Wait for authentication to complete
        print("Authenticating with server...")
        wait_count = 0
        while not auth_completed and wait_count < 8:
            time.sleep(0.25)
            wait_count += 1
        
        # If authentication didn't complete in time, still proceed but warn the user
        if not auth_completed:
            print("⚠️ Authentication response not received yet, but proceeding...")
        
        # Keep the connection alive and allow user to send shabad dynamically
        while True:
            print("\nEnter Shabad and Verse IDs separated by space (or 'exit' to quit):")
            user_input = input("Shabad ID Verse ID: ").strip()
            
            if user_input.lower() == "exit":
                break
                
            # Split the input by space to get shabad_id and verse_id
            parts = user_input.split()
            if len(parts) != 2:
                print("⚠️ Please enter both Shabad ID and Verse ID separated by space (e.g., '1 5')")
                continue
                
            # Use the send_shabad function
            send_shabad(parts[0], parts[1])

        print("🔹 Exiting... disconnecting")
        sio.disconnect()

    except Exception as e:
        print("⚠️ Exception:", e)

if __name__ == "__main__":
    main()
