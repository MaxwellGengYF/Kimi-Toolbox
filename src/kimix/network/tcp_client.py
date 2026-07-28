#!/usr/bin/env python3
"""TCP Socket Client - Low-level TCP socket handling.

This module provides the basic TCP socket infrastructure for connecting to a
TCP server and handling low-level I/O operations.
"""

import socket
import threading
import time
from typing import Optional, Callable

from kimix.ui.printing import print  # noqa: F811 - use ui.printing.print for flush support


class TCPClient:
    """Low-level TCP socket client.

    Handles socket creation, connection, and basic message management.
    Uses length-prefixed message protocol for reliable message framing.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._lock = threading.Lock()
        self._receive_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_connect: Optional[Callable[[], None]] = None
        self._on_disconnect: Optional[Callable[[], None]] = None
        self._on_message: Optional[Callable[[str], None]] = None

    def on_connect(self, callback: Callable[[], None]) -> None:
        """Set callback for successful connection."""
        self._on_connect = callback

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Set callback for disconnection."""
        self._on_disconnect = callback

    def on_message(self, callback: Callable[[str], None]) -> None:
        """Set callback for received messages (as string)."""
        self._on_message = callback

    def connect(self, blocking: bool = False) -> bool:
        """Connect to the server.

        Args:
            blocking: If True, blocks until disconnect() is called.
                     If False, runs receive loop in a background thread.

        Returns:
            True if connection was successful, False otherwise.
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)
        except Exception as e:
            print(f"[TCPClient] Connection error: {e}", flush=True)
            self._socket = None
            return False

        self._running = True
        print(f"[TCPClient] Connected to {self.host}:{self.port}", flush=True)

        if self._on_connect:
            try:
                self._on_connect()
            except Exception as e:
                print(f"[TCPClient] Connect callback error: {e}", flush=True)

        if blocking:
            self._receive_loop()
        else:
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

        return True

    def _receive_loop(self) -> None:
        """Main loop for receiving messages from the server."""
        while self._running and self._socket is not None:
            try:
                # Receive length prefix (4 bytes, network byte order)
                length_bytes = self._recv_all(4)
                if length_bytes is None:
                    break

                length = int.from_bytes(length_bytes, "big")

                # Sanity check
                if length == 0 or length > 10 * 1024 * 1024:  # Max 10MB
                    error_msg = f"Invalid message length: {length}"
                    print(f"[TCPClient] {error_msg}", flush=True)
                    continue

                # Receive payload
                payload = self._recv_all(length)
                if payload is None:
                    break

                message = payload.decode("utf-8")
                # Notify message callback
                if self._on_message:
                    try:
                        self._on_message(message)
                    except Exception as e:
                        print(f"[TCPClient] Message callback error: {e}", flush=True)

            except socket.timeout:
                continue
            except Exception as e:
                error_msg = f"Receive error: {e}"
                print(f"[TCPClient] {error_msg}", flush=True)
                break

        # Disconnected
        self._handle_disconnect()

    def _recv_all(self, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes from the server."""
        if self._socket is None:
            return None

        data = b""
        while len(data) < size:
            try:
                chunk = self._socket.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except Exception:
                return None

        return data

    def send(self, message: str) -> bool:
        """Send a message to the server using length-prefixed protocol."""
        with self._lock:
            if self._socket is None:
                return False

            try:
                payload = message.encode("utf-8")
                length = len(payload)

                # Send length prefix
                length_bytes = length.to_bytes(4, "big")
                self._socket.sendall(length_bytes)

                # Send payload
                self._socket.sendall(payload)
                return True

            except Exception as e:
                print(f"[TCPClient] Send error: {e}", flush=True)
                return False

    def send_bytes(self, data: bytes) -> bool:
        """Send raw bytes to the server using length-prefixed protocol."""
        with self._lock:
            if self._socket is None:
                return False

            try:
                length = len(data)

                # Send length prefix
                length_bytes = struct.pack("!I", length)
                self._socket.sendall(length_bytes)

                # Send payload
                self._socket.sendall(data)

                return True

            except Exception as e:
                print(f"[TCPClient] Send error: {e}", flush=True)
                return False

    def disconnect(self) -> None:
        """Disconnect from the server and close the socket."""
        self._running = False

        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None

        # Wait for receive thread to finish
        if self._receive_thread is not None and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=2.0)

        print("[TCPClient] Disconnected", flush=True)

    def _handle_disconnect(self) -> None:
        """Handle server disconnection."""
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None

        print("[TCPClient] Server disconnected", flush=True)

        if self._on_disconnect:
            try:
                self._on_disconnect()
            except Exception as e:
                print(f"[TCPClient] Disconnect callback error: {e}", flush=True)

    def is_connected(self) -> bool:
        """Check if connected to the server."""
        with self._lock:
            return self._socket is not None

    def wait_for_connection(self, timeout: float = 5.0) -> bool:
        """Wait for connection to be established."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_connected():
                return True
            time.sleep(0.05)
        return False

    def wait_for_disconnection(self, timeout: float = 5.0) -> bool:
        """Wait for disconnection from the server."""
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_connected():
                return True
            time.sleep(0.05)
        return False
