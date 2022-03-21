import math
from typing import Generator


def zero_padded_iterator(
    start: int, end: int, num_digits: int
) -> Generator[str, None, None]:
    """
    A zero-padded integer iterator. The yielded outputs are strings
    of a fixed length, with zero-padding if the numbers themselves
    are too short, e.g., `"00051"`, `"00052"`.

    Args:
        start (int): The integer to start counting from.
        end (int): The integer to stop at. Note that the
            final yielded number is `end-1`.
        num_digits (int): The number of digits for each yield.

    Raises:
        ValueError: If the input limits are incompatible.

    Yields:
        str: Zero-padded integers, e.g., `"00051"`, `"00052"`.
    """

    # Sanity checks
    if start < 0:
        raise ValueError("Must start at a positive integer.")
    if num_digits < 1:
        raise ValueError("Number of digits must be a positive integer.")
    if end < start:
        raise ValueError("End must not be smaller than start.")
    if len(str(start)) > num_digits:
        raise ValueError(
            "Start must not have more digits than the specified number of digits."
        )
    if len(str(end)) > num_digits:
        raise ValueError(
            "End must not have more digits than the specified number of digits."
        )

    count_digits = lambda x: math.floor(math.log10(x)) + 1 if x != 0 else 1

    for i in range(start, end):
        yield "0" * (num_digits - count_digits(i)) + str(i)
