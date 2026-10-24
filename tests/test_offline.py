"""The suite runs without a network, and the guard that proves it works."""

import socket

import pytest

from tests.conftest import NetworkUsed


def test_a_connection_to_another_machine_fails_the_test():
    # 192.0.2.1 is reserved for documentation (RFC 5737): nothing answers there.
    with pytest.raises(NetworkUsed, match="192.0.2.1"):
        socket.create_connection(("192.0.2.1", 443), timeout=1)


def test_this_machine_is_still_reachable():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        socket.create_connection(server.getsockname(), timeout=1).close()
    finally:
        server.close()
