#!/usr/bin/env python3
"""
TCP Socket Server - Low-level TCP socket handling.

This module provides the basic TCP socket infrastructure for accepting connections
and handling low-level I/O operations.
"""

import socket
import threading
import time
from typing import Optional, Callable, Tuple

from kimix.ui.printing import print  # noqa: F811 - use ui.printing.print for flush support


class TCPServer:
    """
    Low-level TCP socket server.
    
    Handles socket creation, binding, listening, and basic connection management.
    Uses length-prefixed message protocol for reliable message framing.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._client_addr: Optional[tuple[str, int]] = None
        self._running = False
        self._lock = threading.Lock()
        self._receive_thread: Optional[threading.Thread] = None
        self._main_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_connect: Optional[Callable[[], None]] = None
        self._on_disconnect: Optional[Callable[[], None]] = None
        self._on_message: Optional[Callable[[str], None]] = None
        self._on_raw_data: Optional[Callable[[bytes], Optional[str]]] = None
    
    def on_connect(self, callback: Callable[[], None]) -> None:
        """Set callback for client connection."""
        self._on_connect = callback
    
    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Set callback for client disconnection."""
        self._on_disconnect = callback
    
    def on_message(self, callback: Callable[[str], None]) -> None:
        """Set callback for received messages (as string)."""
        self._on_message = callback
    
    def on_raw_data(self, callback: Callable[[bytes], Optional[str]]) -> None:
        """Set callback for raw data processing. Returns response string or None."""
        self._on_raw_data = callback
    
    def start(self, blocking: bool = False) -> None:
        """
        Start the server.
        
        Args:
            blocking: If True, blocks until stop() is called.
                     If False, runs in a background thread.
        """
        if blocking:
            self._run_server()
        else:
            self._main_thread = threading.Thread(target=self._run_server, daemon=True)
            self._main_thread.start()
    
    def _run_server(self) -> None:
        """Main server loop."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.settimeout(1.0)  # Allow periodic checks for shutdown
        self._socket.bind((self.host, self.port))
        self._socket.listen(1)
        
        self._running = True
        print(f"[TCPServer] Listening on {self.host}:{self.port}", flush=True)
        
        while self._running:
            try:
                client_sock, client_addr = self._socket.accept()
                self._handle_new_client(client_sock, client_addr)
            except socket.timeout:
                continue
            except OSError:
                break
        
        print("[TCPServer] Stopped listening", flush=True)
    
    def _handle_new_client(self, client_sock: socket.socket, client_addr: tuple[str, int]) -> None:
        """Handle a new client connection."""
        with self._lock:
            # Close existing client if any
            if self._client_socket is not None:
                print(f"[TCPServer] New client from {client_addr}, closing old connection", flush=True)
                try:
                    self._client_socket.close()
                except:
                    pass
            
            self._client_socket = client_sock
            self._client_addr = client_addr
            self._client_socket.settimeout(5.0)
        
        print(f"[TCPServer] Client connected from {client_addr}", flush=True)
        
        if self._on_connect:
            try:
                self._on_connect()
            except Exception as e:
                print(f"[TCPServer] Connect callback error: {e}", flush=True)
        
        # Start receive thread
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
    
    def _receive_loop(self) -> None:
        """Main loop for receiving messages from the client."""
        while self._running and self._client_socket is not None:
            try:
                # Receive length prefix (4 bytes, network byte order)
                length_bytes = self._recv_all(4)
                if length_bytes is None:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # Sanity check
                if length == 0 or length > 10 * 1024 * 1024:  # Max 10MB
                    continue
                
                # Receive payload
                payload = self._recv_all(length)
                if payload is None:
                    break
                
                message = payload.decode('utf-8', errors='replace')
                
                # Notify message callback
                if self._on_message:
                    try:
                        self._on_message(message)
                    except Exception as e:
                        print(f"[TCPServer] Message callback error: {e}", flush=True)
                
                # Process through raw data handler if set
                if self._on_raw_data:
                    try:
                        response = self._on_raw_data(payload)
                        if response:
                            self.send(response)
                    except Exception as e:
                        print(f"[TCPServer] Raw data handler error: {e}", flush=True)
                
            except socket.timeout:
                continue
            except Exception:
                break
        
        # Client disconnected
        self._handle_disconnect()
    
    def _recv_all(self, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes from the client."""
        if self._client_socket is None:
            return None
        
        data = b''
        while len(data) < size:
            try:
                chunk = self._client_socket.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except Exception:
                return None
        
        return data
    
    def send(self, message: str) -> bool:
        """Send a message to the connected client using length-prefixed protocol."""
        with self._lock:
            if self._client_socket is None:
                return False
            
            try:
                payload = message.encode('utf-8')
                length = len(payload)
                
                # Send length prefix
                length_bytes = length.to_bytes(4, 'big')
                self._client_socket.sendall(length_bytes)
                
                # Send payload
                self._client_socket.sendall(payload)
                
                # print(f"[TCPServer] Sent: {message[:100]}{'...' if len(message) > 100 else ''}", flush=True)
                return True
                
            except Exception:
                return False
    
    def send_bytes(self, data: bytes) -> bool:
        """Send raw bytes to the connected client using length-prefixed protocol."""
        with self._lock:
            if self._client_socket is None:
                return False
            
            try:
                length = len(data)
                
                # Send length prefix
                length_bytes = struct.pack('!I', length)
                self._client_socket.sendall(length_bytes)
                
                # Send payload
                self._client_socket.sendall(data)
                
                return True
                
            except Exception:
                return False
    
    def disconnect_client(self) -> None:
        """Force disconnect the current client."""
        with self._lock:
            if self._client_socket is not None:
                try:
                    self._client_socket.close()
                except:
                    pass
                self._client_socket = None
        print("[TCPServer] Client forcefully disconnected", flush=True)
    
    def _handle_disconnect(self) -> None:
        """Handle client disconnection."""
        with self._lock:
            if self._client_socket is not None:
                try:
                    self._client_socket.close()
                except:
                    pass
                self._client_socket = None
                self._client_addr = None
        
        print("[TCPServer] Client disconnected", flush=True)
        
        if self._on_disconnect:
            try:
                self._on_disconnect()
            except Exception as e:
                print(f"[TCPServer] Disconnect callback error: {e}", flush=True)
    
    def is_client_connected(self) -> bool:
        """Check if a client is currently connected."""
        with self._lock:
            return self._client_socket is not None
    
    def wait_for_connection(self, timeout: float = 5.0) -> bool:
        """Wait for a client to connect."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_client_connected():
                return True
            time.sleep(0.05)
        return False
    
    def wait_for_disconnection(self, timeout: float = 5.0) -> bool:
        """Wait for the client to disconnect."""
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_client_connected():
                return True
            time.sleep(0.05)
        return False
    
    def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False
        
        self.disconnect_client()
        
        if self._socket is not None:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
        
        # Wait for threads to finish
        if self._receive_thread is not None and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)
        
        if self._main_thread is not None and self._main_thread.is_alive():
            self._main_thread.join(timeout=2.0)
        
        print("[TCPServer] Stopped", flush=True)
    
    def get_client_addr(self) -> Optional[tuple[str, int]]:
        """Get the address of the connected client."""
        with self._lock:
            return self._client_addr
