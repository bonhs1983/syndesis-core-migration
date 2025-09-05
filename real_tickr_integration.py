"""
Real TICKR A204 Bluetooth Integration Module
Connects to actual TICKR device via BLE and reads authentic HRV data
"""

import asyncio
import json
import time
from bleak import BleakScanner, BleakClient
import struct
import logging
from typing import Optional, Dict, Any

# TICKR A204 Bluetooth Configuration
TICKR_DEVICE_NAME = "TICKR A204"
HEART_RATE_SERVICE_UUID = "0000180D-0000-1000-8000-00805F9B34FB"
HEART_RATE_MEASUREMENT_UUID = "00002A37-0000-1000-8000-00805F9B34FB"

class RealTICKRIntegration:
    def __init__(self):
        self.client: Optional[BleakClient] = None
        self.device = None
        self.is_connected = False
        self.last_hrv = None
        self.last_hr = None
        self.rr_intervals = []
        self.coherence_score = None
        
    async def scan_for_tickr(self) -> Optional[str]:
        """Scan for TICKR A204 device and return MAC address"""
        print("🔍 Scanning for TICKR A204 device...")
        
        devices = await BleakScanner.discover(timeout=10.0)
        
        for device in devices:
            if device.name and "TICKR" in device.name:
                print(f"✅ Found TICKR device: {device.name} ({device.address})")
                return device.address
                
        print("❌ No TICKR device found")
        return None
    
    async def connect_to_tickr(self, mac_address: str) -> bool:
        """Connect to TICKR device via BLE"""
        try:
            print(f"🔗 Connecting to TICKR at {mac_address}...")
            
            self.client = BleakClient(mac_address)
            await self.client.connect()
            
            if self.client.is_connected:
                print("✅ Connected to TICKR A204")
                self.is_connected = True
                
                # Start heart rate notifications
                await self.client.start_notify(
                    HEART_RATE_MEASUREMENT_UUID, 
                    self._heart_rate_handler
                )
                
                return True
            else:
                print("❌ Failed to connect to TICKR")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def _heart_rate_handler(self, sender, data: bytearray):
        """Handle incoming heart rate data from TICKR"""
        try:
            # Parse BLE Heart Rate Measurement format
            flags = data[0]
            
            if flags & 0x01:  # 16-bit HR value
                hr = struct.unpack('<H', data[1:3])[0]
                offset = 3
            else:  # 8-bit HR value
                hr = data[1]
                offset = 2
            
            # Extract RR intervals if present
            rr_intervals = []
            if flags & 0x10 and len(data) > offset:  # RR intervals present
                rr_data = data[offset:]
                for i in range(0, len(rr_data), 2):
                    if i + 1 < len(rr_data):
                        rr = struct.unpack('<H', rr_data[i:i+2])[0]
                        rr_ms = rr * 1000 / 1024  # Convert to milliseconds
                        rr_intervals.append(rr_ms)
            
            # Update data
            self.last_hr = hr
            self.rr_intervals.extend(rr_intervals)
            
            # Keep only last 20 RR intervals for HRV calculation
            if len(self.rr_intervals) > 20:
                self.rr_intervals = self.rr_intervals[-20:]
            
            # Calculate HRV (RMSSD)
            if len(self.rr_intervals) >= 2:
                diffs = [abs(self.rr_intervals[i] - self.rr_intervals[i-1]) 
                        for i in range(1, len(self.rr_intervals))]
                rmssd = (sum(d**2 for d in diffs) / len(diffs)) ** 0.5
                self.last_hrv = rmssd
                
                # Calculate coherence score
                self._calculate_coherence()
            
            print(f"💓 HR: {hr} bpm | HRV: {self.last_hrv:.1f}ms | Coherence: {self.coherence_score:.1f}%")
            
        except Exception as e:
            print(f"❌ Error parsing heart rate data: {e}")
    
    def _calculate_coherence(self):
        """Calculate HRV coherence score"""
        if len(self.rr_intervals) < 5:
            self.coherence_score = 0
            return
            
        # Simple coherence calculation based on RR interval variability
        intervals = self.rr_intervals[-10:]  # Use last 10 intervals
        mean_rr = sum(intervals) / len(intervals)
        variance = sum((rr - mean_rr)**2 for rr in intervals) / len(intervals)
        
        # Coherence is inversely related to excessive variability
        # Higher coherence = more rhythmic, less chaotic
        if variance == 0:
            self.coherence_score = 100
        else:
            # Normalize to 0-100% scale
            coherence = max(0, min(100, 100 - (variance / 1000)))
            self.coherence_score = coherence
    
    async def disconnect(self):
        """Disconnect from TICKR device"""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("🔌 Disconnected from TICKR")
        
        self.is_connected = False
        self.client = None
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current authentic TICKR data"""
        return {
            "connected": self.is_connected,
            "heart_rate": self.last_hr,
            "hrv": self.last_hrv,
            "coherence": self.coherence_score,
            "rr_intervals": self.rr_intervals.copy(),
            "timestamp": time.time()
        }

# Global TICKR instance
tickr = RealTICKRIntegration()

async def initialize_tickr():
    """Initialize and connect to TICKR device"""
    # Scan for device
    mac_address = await tickr.scan_for_tickr()
    if not mac_address:
        return False
    
    # Connect to device
    return await tickr.connect_to_tickr(mac_address)

def get_real_tickr_data():
    """Get real TICKR data synchronously"""
    return tickr.get_current_data()

if __name__ == "__main__":
    # Test connection
    async def test():
        success = await initialize_tickr()
        if success:
            print("✅ TICKR connection test successful")
            # Keep reading for 30 seconds
            await asyncio.sleep(30)
            await tickr.disconnect()
        else:
            print("❌ TICKR connection test failed")
    
    asyncio.run(test())