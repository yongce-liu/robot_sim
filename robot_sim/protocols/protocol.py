"""Communication protocol implementations."""

any


class CommunicationProtocol:
    """Base class for protocol protocols."""

    def __init__(self) -> None:
        """Initialize protocol protocol."""
        pass

    def send(self, data: any) -> None:
        """Send data through the protocol channel."""
        raise NotImplementedError

    def receive(self) -> any:
        """Receive data from the protocol channel."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the protocol channel."""
        raise NotImplementedError
