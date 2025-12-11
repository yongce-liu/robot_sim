"""Communication protocol implementations."""

any


class CommunicationProtocol:
    """Base class for communication protocols."""

    def __init__(self) -> None:
        """Initialize communication protocol."""
        pass

    def send(self, data: any) -> None:
        """Send data through the communication channel."""
        raise NotImplementedError

    def receive(self) -> any:
        """Receive data from the communication channel."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the communication channel."""
        raise NotImplementedError
