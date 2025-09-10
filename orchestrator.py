#!/usr/bin/env python3
"""
STTM Agent Orchestrator
-----------------------
This script orchestrates the integration between the ASR-based agent
and the STTM sync controller.
"""

import os
import sys
import argparse
import time

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from sttm_ui_controller import activate_sttm, open_sync_panel, close_sync_panel, extract_sync_code_pin
from sttm_sync_client import STTMSyncClient
from verse_dataset import get_shabad_id_for_verse, verse_dataset

def run_standalone_sync_mode():
    """Run in standalone sync mode - manual control of STTM via sync"""
    print("=== STTM Standalone Sync Mode ===")
    
    # Open sync panel and extract codes
    open_sync_panel()
    
    # Extract sync code and PIN
    sync_code, pin = extract_sync_code_pin()
    print(f"Extracted: code={sync_code}, PIN={pin}")
    
    # Close sync panel
    close_sync_panel()
    
    if not sync_code or not pin:
        print("Failed to extract sync code or PIN")
        return
        
    # Connect to STTM
    sync_client = STTMSyncClient()
    success = sync_client.connect_with_code_pin(sync_code, pin)
    
    if not success:
        print("Failed to connect to STTM")
        return
        
    # Simple CLI for controlling STTM
    print("\n=== STTM Control CLI ===")
    print("Enter 'exit' to quit")
    print("Enter 'verse_id' to display a verse")
    print("Enter 'verse_id shabad_id' to display a verse with explicit shabad ID\n")
    
    try:
        while True:
            cmd = input("STTM> ").strip()
            
            if cmd.lower() in ["exit", "quit"]:
                break
                
            parts = cmd.split()
            if len(parts) == 2:
                # Verse ID and shabad ID
                verse_id = parts[0]
                shabad_id = parts[1]
                
                # Add to mapping
                verse_dataset.add_mapping(verse_id, shabad_id)
                
                sync_client.send_verse(shabad_id, verse_id)
                
            else:
                print("Invalid command format")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        sync_client.disconnect()

def run_full_agent():
    """
    Run the full ASR-based agent with pre-established STTM connection
    
    This function will:
    1. Open sync panel and get code/PIN
    2. Connect to STTM
    3. Start the ASR agent with the connection already established
    """
    print("=== Initializing STTM connection before starting ASR agent ===")
    
    # Connect to STTM first
    # 1. Open sync panel and extract codes
    open_sync_panel()
    
    # 2. Extract sync code and PIN
    sync_code, pin = extract_sync_code_pin()
    print(f"Extracted: code={sync_code}, PIN={pin}")
    
    # 3. Close sync panel
    close_sync_panel()
    
    if not sync_code or not pin:
        print("Failed to extract sync code or PIN. Cannot proceed.")
        return
        
    # 4. Connect to STTM
    from sttm_sync_client import STTMSyncClient
    sync_client = STTMSyncClient()
    success = sync_client.connect_with_code_pin(sync_code, pin)
    
    if not success:
        print("Failed to connect to STTM. Cannot proceed.")
        return
    
    print("Starting ASR agent with pre-established connection...")
    
    # 5. Import agent module
    import agent_full
    
    # 6. Set the pre-established connection in the agent
    agent_full.sttm_sync = sync_client
    
    # 7. Run the agent (this will block)
    agent_full.main()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="STTM Agent Orchestrator")
    parser.add_argument("--mode", choices=["agent", "sync"], default="agent",
                        help="Mode to run (agent=full ASR agent, sync=standalone sync control)")
    
    args = parser.parse_args()
    
    if args.mode == "agent":
        run_full_agent()
    else:
        run_standalone_sync_mode()

if __name__ == "__main__":
    main()
