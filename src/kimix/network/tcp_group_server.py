#!/usr/bin/env python3
"""TCP Group Server - Multi-client TCP socket server using a thread pool.

This module provides a TCP server that can handle multiple simultaneous client
connections. Each client is served by a dedicated thread from a thread pool.
"""

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from kimix.ui.printing import print  # noqa: F811 - use ui.printing.print for flush support


class TcpGroupServer:
    """Multi-client TCP socket server using a thread pool.

    Handles socket creation, binding, listening, and managing multiple client
    connections concurrently. Uses length-prefixed message protocol for reliable
    message framing.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8888, max_workers: int = 10):
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._lock = threading.Lock()
        self._next_client_id = 0
        self._main_thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None

        # client_id -> socket mapping
        self._clients: dict[int, socket.socket] = {}
        # client_id -> address mapping
        self._client_addrs: dict[int, tuple[str, int]] = {}

        # Callbacks
        self._on_client_connect: Optional[Callable[[int, tuple[str, int]], None]] = None
        self._on_client_disconnect: Optional[Callable[[int], None]] = None
        self._on_message: Optional[Callable[[int, str], None]] = None
        self._on_raw_data: Optional[Callable[[int, bytes], Optional[str]]] = None

    def on_client_connect(self, callback: Callable[[int, tuple[str, int]], None]) -> None:
        """Set callback for client connection. Called with (client_id, addr)."""
        self._on_client_connect = callback

    def on_client_disconnect(self, callback: Callable[[int], None]) -> None:
        """Set callback for client disconnection. Called with (client_id)."""
        self._on_client_disconnect = callback

    def on_message(self, callback: Callable[[int, str], None]) -> None:
        """Set callback for received messages. Called with (client_id, message)."""
        self._on_message = callback

    def on_raw_data(self, callback: Callable[[int, bytes], Optional[str]]) -> None:
        """Set callback for raw data processing. Returns response string or None."""
        self._on_raw_data = callback

    def start(self, blocking: bool = False) -> None:
        """Start the server.

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
        self._socket.listen(5)

        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="TcpGroupClient")
        print(f"[TcpGroupServer] Listening on {self.host}:{self.port}", flush=True)

        while self._running:
            try:
                client_sock, client_addr = self._socket.accept()
                self._handle_new_client(client_sock, client_addr)
            except socket.timeout:
                continue
            except OSError:
                break

        print("[TcpGroupServer] Stopped listening", flush=True)

    def _handle_new_client(self, client_sock: socket.socket, client_addr: tuple[str, int]) -> None:
        """Handle a new client connection."""
        with self._lock:
            client_id = self._next_client_id
            self._next_client_id += 1
            self._clients[client_id] = client_sock
            self._client_addrs[client_id] = client_addr
            client_sock.settimeout(5.0)

        print(f"[TcpGroupServer] Client {client_id} connected from {client_addr}", flush=True)

        if self._on_client_connect:
            try:
                self._on_client_connect(client_id, client_addr)
            except Exception as e:
                print(f"[TcpGroupServer] Connect callback error: {e}", flush=True)

        # Submit receive task to thread pool
        if self._executor is not None:
            self._executor.submit(self._receive_loop, client_id)

    def _receive_loop(self, client_id: int) -> None:
        """Main loop for receiving messages from a client."""
        while self._running:
            with self._lock:
                client_sock = self._clients.get(client_id)
            if client_sock is None:
                break

            try:
                # Receive length prefix (4 bytes, network byte order)
                length_bytes = self._recv_all(client_id, 4)
                if length_bytes is None:
                    break

                length = int.from_bytes(length_bytes, "big")

                # Sanity check
                if length == 0 or length > 10 * 1024 * 1024:  # Max 10MB
                    error_msg = f"Invalid message length: {length}"
                    print(f"[TcpGroupServer] Client {client_id}: {error_msg}", flush=True)
                    continue

                # Receive payload
                payload = self._recv_all(client_id, length)
                if payload is None:
                    break

                message = payload.decode("utf-8", errors="replace")

                # Notify message callback
                if self._on_message:
                    try:
                        self._on_message(client_id, message)
                    except Exception as e:
                        print(f"[TcpGroupServer] Message callback error: {e}", flush=True)

                # Process through raw data handler if set
                if self._on_raw_data:
                    try:
                        response = self._on_raw_data(client_id, payload)
                        if response:
                            self.send(client_id, response)
                    except Exception as e:
                        print(f"[TcpGroupServer] Raw data handler error: {e}", flush=True)

            except socket.timeout:
                continue
            except Exception as e:
                error_msg = f"Receive error: {e}"
                print(f"[TcpGroupServer] Client {client_id}: {error_msg}", flush=True)
                break

        # Client disconnected
        self._handle_disconnect(client_id)

    def _recv_all(self, client_id: int, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes from a client."""
        with self._lock:
            client_sock = self._clients.get(client_id)
        if client_sock is None:
            return None

        data = b""
        while len(data) < size:
            try:
                chunk = client_sock.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                continue
            except Exception:
                return None

        return data

    def send(self, client_id: int, message: str) -> bool:
        """Send a message to a specific client using length-prefixed protocol."""
        with self._lock:
            client_sock = self._clients.get(client_id)
            if client_sock is None:
                return False

            try:
                payload = message.encode("utf-8")
                length = len(payload)

                # Send length prefix
                length_bytes = length.to_bytes(4, "big")
                client_sock.sendall(length_bytes)

                # Send payload
                client_sock.sendall(payload)

                return True

            except Exception as e:
                print(f"[TcpGroupServer] Send error to client {client_id}: {e}", flush=True)
                return False

    def send_bytes(self, client_id: int, data: bytes) -> bool:
        """Send raw bytes to a specific client using length-prefixed protocol."""
        with self._lock:
            client_sock = self._clients.get(client_id)
            if client_sock is None:
                return False

            try:
                length = len(data)

                # Send length prefix
                length_bytes = struct.pack("!I", length)
                client_sock.sendall(length_bytes)

                # Send payload
                client_sock.sendall(data)

                return True

            except Exception as e:
                print(f"[TcpGroupServer] Send error to client {client_id}: {e}", flush=True)
                return False

    def broadcast(self, message: str) -> dict[int, bool]:
        """Broadcast a message to all connected clients.

        Returns:
            dict mapping client_id to success bool.
        """
        with self._lock:
            client_ids = list(self._clients.keys())

        results: dict[int, bool] = {}
        for cid in client_ids:
            results[cid] = self.send(cid, message)
        return results

    def broadcast_bytes(self, data: bytes) -> dict[int, bool]:
        """Broadcast raw bytes to all connected clients.

        Returns:
            dict mapping client_id to success bool.
        """
        with self._lock:
            client_ids = list(self._clients.keys())

        results: dict[int, bool] = {}
        for cid in client_ids:
            results[cid] = self.send_bytes(cid, data)
        return results

    def disconnect_client(self, client_id: int) -> bool:
        """Force disconnect a specific client."""
        with self._lock:
            client_sock = self._clients.get(client_id)
            if client_sock is None:
                return False

            try:
                client_sock.close()
            except Exception:
                pass

        print(f"[TcpGroupServer] Client {client_id} forcefully disconnected", flush=True)
        return True

    def _handle_disconnect(self, client_id: int) -> None:
        """Handle client disconnection."""
        with self._lock:
            if client_id in self._clients:
                try:
                    self._clients[client_id].close()
                except Exception:
                    pass
                del self._clients[client_id]
                del self._client_addrs[client_id]

        print(f"[TcpGroupServer] Client {client_id} disconnected", flush=True)

        if self._on_client_disconnect:
            try:
                self._on_client_disconnect(client_id)
            except Exception as e:
                print(f"[TcpGroupServer] Disconnect callback error: {e}", flush=True)

    def get_client_ids(self) -> list[int]:
        """Get list of currently connected client IDs."""
        with self._lock:
            return list(self._clients.keys())

    def get_client_addr(self, client_id: int) -> Optional[tuple[str, int]]:
        """Get the address of a specific client."""
        with self._lock:
            return self._client_addrs.get(client_id)

    def is_client_connected(self, client_id: int) -> bool:
        """Check if a specific client is currently connected."""
        with self._lock:
            return client_id in self._clients

    def get_client_count(self) -> int:
        """Get the number of currently connected clients."""
        with self._lock:
            return len(self._clients)

    def wait_for_clients(self, count: int, timeout: float = 5.0) -> bool:
        """Wait until at least 'count' clients are connected."""
        start = time.time()
        while time.time() - start < timeout:
            if self.get_client_count() >= count:
                return True
            time.sleep(0.05)
        return False

    def wait_for_client_disconnect(self, client_id: int, timeout: float = 5.0) -> bool:
        """Wait for a specific client to disconnect."""
        start = time.time()
        while time.time() - start < timeout:
            if not self.is_client_connected(client_id):
                return True
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False

        with self._lock:
            for client_id, client_sock in list(self._clients.items()):
                try:
                    client_sock.close()
                except Exception:
                    pass
            self._clients.clear()
            self._client_addrs.clear()

        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Wait for main thread to finish
        if self._main_thread is not None and self._main_thread.is_alive():
            self._main_thread.join(timeout=2.0)

        print("[TcpGroupServer] Stopped", flush=True)
