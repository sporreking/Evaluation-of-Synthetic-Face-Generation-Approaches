import socket
import struct
from typing import List

CHUNK_SIZE = 256 * 1024
DEFAULT_LATENT_CODE_PORT = 6969


def receive_latent_codes(num_floats_per_latent_code: int) -> List[float]:
    """
    Receives latent codes using a socket client.

    Args:
        num_floats_per_latent_code (int): Number of floats per latent code.

    Returns:
        list: All latent codes received.
    """
    print("Connecting...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, CHUNK_SIZE)
    connected = False
    while not connected:
        try:
            s.connect(("localhost", DEFAULT_LATENT_CODE_PORT))
        except ConnectionRefusedError:
            continue
        connected = True

    print("Receiving dimensions...")
    num_latent_codes = int.from_bytes(s.recv(4), "big")

    print("Receiving data...")
    data = []
    while True:
        p = s.recv(CHUNK_SIZE)
        if len(p) == 0:
            break
        data.append(p)
    data = b"".join(data)

    print("Decoding data...")
    latent_codes = struct.unpack(
        f"<{num_latent_codes*num_floats_per_latent_code}d", data
    )

    print("Done! Closing stream.")
    s.close()
    return latent_codes
