from typing import Any, Iterable, Union
from pathlib import Path
import re

ROOT_DIR = Path(".prompt_util")

_indent_modes = [
    (None, "", None),  # 0
    ("|" + "-" * 10, "| ", "|" + "-" * 10),  # 1
    ("?" * 10, "? ", "?" * 10),  # 2
    ("!" * 10, "! ", "!" * 10),  # 3
]

_indent_mode_stack = [0]

_line_broken = True


def _get_indent() -> str:
    return "".join([_indent_modes[i][1] for i in _indent_mode_stack])


def _print(obj: Any = "", new_line: bool = True) -> None:
    global _line_broken
    text = str(obj).replace("\n", "\n" + _get_indent())
    if _line_broken:
        print(_get_indent(), end="")
    print(text, end="\n" if new_line else "")
    _line_broken = new_line


def _input(obj: Any = "Press Enter to continue") -> str:
    return input(_get_indent() + str(obj) + " > ")


def _load_strings(file: Union[Path, str]) -> list[str]:
    if (ROOT_DIR / file).is_file():
        with open(ROOT_DIR / file, "r") as f:
            return [s[:-1] for s in f.readlines()]
    else:
        return None


def _save_strings(file: Union[Path, str], strings: Iterable[str]) -> None:
    if not ROOT_DIR.is_dir():
        ROOT_DIR.mkdir()

    with open(ROOT_DIR / file, "w") as f:
        f.writelines([s + "\n" for s in strings])


def _opt_match(
    option: str, options: Iterable[str], case_sensitive: bool = False
) -> bool:
    return (
        option in options
        if case_sensitive
        else option.lower() in (opt.lower() for opt in options)
    )


def _prompt_binary(
    directive: str,
    true_option: str = "yes",
    false_option: str = "no",
    short_answers: bool = True,
    case_sensitive: bool = False,
) -> bool:

    push_indent(1)

    # Prompt user for answer
    answer: str = None
    while answer is None or not _opt_match(
        answer,
        [
            *((true_option[0], false_option[0]) if short_answers else []),
            true_option,
            false_option,
        ],
        case_sensitive,
    ):
        opt_prompt = "/".join(
            (true_option[0], false_option[0])
            if short_answers
            else (true_option, false_option)
        )
        answer = _input(f"{directive} ({opt_prompt})")

    pop_indent()

    return _opt_match(answer, (true_option[0], true_option), case_sensitive)


def _print_opt_list(options: Iterable[str], indices: Iterable[int] = None) -> None:
    if indices is None:
        for i, opt in enumerate(options):
            _print(f"  {i}: {opt}")
    elif len(options) == len(indices):
        for i, index in enumerate(indices):
            _print(f"  {index}: {options[i]}")
    else:
        raise ValueError("Every option must have a unique index.")


def _unravel_indices(index_string: str) -> list[int]:

    # Remove blankspaces
    index_string = index_string.replace(" ", "")
    if index_string.isdigit():  # "5", "69" etc.
        return [int(index_string)]
    elif re.match(r"^\d+-\d+$", index_string):  # "0-5", "2-9" etc.
        split = index_string.split("-")
        return list(range(int(split[0]), int(split[1]) + 1))
    elif re.match(r"^\d+:\d+$", index_string):  # "0:5", "2:9" etc.
        split = index_string.split(":")
        return list(range(int(split[0]), int(split[1]) + 1))
    elif re.match(
        r"^(((\d+(:|-)\d+)|\d+),)*((\d+(:|-)\d+)|\d+)$", index_string
    ):  # "1,3-9,13:17" etc.
        sections = index_string.split(",")
        output = []
        for section in sections:
            output += _unravel_indices(section)
        return output
    else:  # Invalid format
        return []


def input_float(directive: str, min: float = None, max: float = None) -> float:
    """
    Ask the user for float input.

    Args:
        directive (str): The directive to prompt the user with.
        min (float, optional): Minimum allowed value for the input. Set to
            `None` if there should not be a lower limit. Defaults to None.
        max (float, optional): Maximum allowed value for the input. Set to
            `None` if there should not be a upper limit. Defaults to None.

    Returns:
        float: The user input.
    """
    push_indent(1)

    answer = None
    while (
        answer is None
        or not answer.isnumeric()
        or (min is not None and float(answer) < min)
        or (max is not None and float(answer) > max)
    ):
        answer = _input(directive)

    pop_indent()

    return int(answer)


def input_int(directive: str, min: int = None, max: int = None) -> int:
    """
    Ask the user for integer input.

    Args:
        directive (str): The directive to prompt the user with.
        min (int, optional): Minimum allowed value for the input. Set to
            `None` if there should not be a lower limit. Defaults to None.
        max (int, optional): Maximum allowed value for the input. Set to
            `None` if there should not be a upper limit. Defaults to None.

    Returns:
        int: The user input.
    """
    push_indent(1)

    answer = None
    while (
        answer is None
        or not (
            answer.isdigit()
            or (len(answer) > 1 and answer[0] == "-" and answer[1:].isdigit())
        )
        or (min is not None and int(answer) < min)
        or (max is not None and int(answer) > max)
    ):
        answer = _input(directive)

    pop_indent()

    return int(answer)


def input_string(directive: str) -> str:
    """
    Ask the use for string input.

    Args:
        directive (str): The directive to prompt the user with.

    Returns:
        str: The text input.
    """
    return _input(directive)


def input_type(directive: str, t: type) -> Any:
    """
    Ask the user for input of type `t`.

    Args:
        directive (str): The directive to prompt the user with.
        t (type): The input type to receive. Supported input
            types are: `float`, `int`, and `str`.

    Raises:
        ValueError: If an unsupported input type was requested.

    Returns:
        Any: The input of type `t`.
    """
    if t == int:
        return input_int(directive)
    elif t == float:
        return input_float(directive)
    elif t == str:
        return input_string(directive)
    else:
        raise ValueError(f"Cannot handle input of type: '{t}'")


def input_continue() -> None:
    """
    Asks the user to press 'Enter' to continue.
    """
    _input()


def print_with_border(obj: Any, symbol: str = "#", side_symbol: str = None) -> None:
    """
    Prints the specified `obj` inside of a border.

    Args:
        obj (Any): The object to print (it will be cast to a string).
        symbol (str, optional): The symbol to use for the border. Defaults to "#".
        side_symbol (str, optional): The symbol to use for the sides of the border,
            or `None` if it should be the same as `symbol`. Defaults to None.

    Raises:
        ValueError: If `symbol` is more than one character long.
    """
    if len(symbol) != 1:
        raise ValueError(
            f"Symbol must be exactly one character long. Invalid: '{symbol}'."
        )

    if side_symbol is None:
        side_symbol = symbol

    _print(symbol * (len(str(obj)) + 2 * (len(side_symbol) + 1)))
    _print(side_symbol + " " + str(obj) + " " + side_symbol)
    _print(symbol * (len(str(obj)) + 2 * (len(side_symbol) + 1)))


def push_indent(mode: int) -> None:
    """
    When an indent is pushed, all following prints issued by the PromptUtil
    will appear at an indent, until `pop_indent()` is called.

    Note that indents are pushed onto a stack, such that it is possible to
    have multiple layers of indents with different esthetics.

    Args:
        mode (int): The indent theme to use. This determines what the
            indent will look like.

    Raises:
        ValueError: If an invalid mode was selected.
    """
    if mode < 0 or mode >= len(_indent_modes):
        raise ValueError(
            f"Invalid indent mode! Must be integer in range 0-{len(_indent_modes)}."
        )

    preborder = _indent_modes[mode][0]

    if preborder is not None:
        _print(preborder)

    _indent_mode_stack.append(mode)


def pop_indent() -> None:
    """
    Pops the latest indent layer that was pushed with `push_indent(mode)`.

    Raises:
        ValueError: If no indents have been pushed, i.e., the only
            remaining level is root.
    """
    global _line_broken

    if len(_indent_mode_stack) <= 1:
        raise ValueError("Tried to pop first indent level!")

    postborder = _indent_modes[_indent_mode_stack.pop()][2]

    if postborder is not None:
        if not _line_broken:
            _line_broken = True
            print()
        _print(postborder)

    _print()


def tablify(columns: Iterable[Union[Iterable[str], str]]) -> list[str]:
    """
    Derives a list of table rows, based on the supplied `columns`.

    Args:
        columns (Iterable[Union[Iterable[str], str]]): The columns to use.
            All columns must be of same length, or a single string. If a
            single string is encountered, it will be repeated for the
            entire column such that it adds upp to the other column lengths.

    Returns:
        list[str]: The rows of the table.
    """
    column_length: int
    for column in columns:
        if not isinstance(column, str):
            column_length = len(column)
            break

    assert all(
        isinstance(column, str) or len(column) == column_length for column in columns
    )

    max_lengths = [
        (
            len(column)
            if isinstance(column, str)
            else max([len(item) for item in column])
        )
        for column in columns
    ]

    pad = lambda text, column_index: text + " " * (
        max_lengths[column_index] - len(text)
    )

    return [
        " | ".join(
            [
                pad(column if isinstance(column, str) else column[i], j)
                for j, column in enumerate(columns)
            ]
        )
        for i in range(column_length)
    ]


def prompt_yes_no(directive: str, short_answers: bool = True) -> bool:
    """
    Gives the user a `directive`, and asks them to enter "yes" or "no".

    Args:
        directive (str): The directive to give the user.
        short_answers (bool, optional): If `True`, the user is allowed
            to enter "y" or "n" in addition to the full answers. Defaults to True.

    Returns:
        bool: `True` if the user replied with "yes" (or "y").
    """
    return _prompt_binary(directive, "yes", "no", short_answers=short_answers)


def prompt_options(
    directive: str,
    options: Iterable[Any],
    default_index: int = 0,
    return_index: bool = False,
) -> Union[Any, int]:
    """
    Prompts the user with a list of `options`, each with their own index.

    The user is asked to pick one of these options by enterin its index.

    Args:
        directive (str): The directive to give the user.
        options (Iterable[Any]): The options to pick from.
        default_index (int, optional): The index of the option that will
            be picked if the user enters an empty string. Defaults to 0.
        return_index (bool, optional): If `True`, the index of the option
            that was picked will be returned, rather than the option itself.
                Defaults to False.

    Raises:
        ValueError: If `default_index` does not refer to a valid option, i.e.,
            the value is out of index range.

    Returns:
        Union[Any, int]: The chosen option, or its index if `return_index=True`.
    """
    push_indent(1)

    if default_index < 0 or default_index >= len(options):
        raise ValueError(
            f"Default index must refer to a valid option (0-{len(options)-1})."
        )

    answer: str = None
    while (
        answer is None
        or not answer.isdigit()
        or int(answer) < 0
        or int(answer) >= len(options)
    ) and answer != "":
        _print(directive)
        _print_opt_list(options)
        answer = _input(f"(default: {default_index})")

    if answer == "":
        answer = default_index

    _print(f"* Selection: {options[int(answer)]}")

    pop_indent()

    return int(answer) if return_index else options[int(answer)]


def print_list(
    header: str = None,
    items: Iterable[Any] = [],
    bullet_symbol: str = "*",
    header_border_symbol: str = None,
    header_border_side_symbol: str = None,
) -> None:
    """
    Prints a list with a header.

    Args:
        header (str, optional): The header of the list. Defaults to None.
        items (Iterable[Any], optional): The list items. Defaults to [].
        bullet_symbol (str, optional): The symbol to use for bullets. Defaults to "*".
        header_border_symbol (str, optional): The symbol to use for the header border,
            or `None` if there should be no border. Defaults to None.
        header_border_side_symbol (str, optional): The symbol to use for the sides of
            the header border, or `None` if it should be the same as `header_border_symbol`.
            Defaults to None.
    """
    push_indent(1)
    if header is not None:
        if header_border_symbol is not None:
            print_with_border(header, header_border_symbol, header_border_side_symbol)
        else:
            _print(header)
    for item in items:
        _print(bullet_symbol + " " + str(item))
    pop_indent()


def print_with_indent(obj: Any = "", new_line: bool = True) -> None:
    """
    Performs a regular print with indentation according to the current
    indent stack. See `push_indent(mode)` and `pop_indent()` for more information.

    Args:
        obj (Any, optional): The object to print (it will be cast to string).
            Defaults to "".
        new_line (bool, optional): If `True`, a newline character will be appended. Defaults to True.
    """
    _print(obj, new_line)


def prompt_multi_options(
    directive: str,
    options: Iterable[Any],
    default_indices: Iterable[int] = [],
    default_file: Union[Path, str] = None,
    return_index: bool = False,
    allow_empty: bool = False,
) -> Union[list[Any], list[int]]:
    """
    Prompt the user with a `directive`, allowing them to pick multiple options.

    The user adds / removes items from their selection until they are satisfied.

    Args:
        directive (str): The directive to give the user.
        options (Iterable[Any]): The options to choose from.
        default_indices (Iterable[int], optional): Indices of items that should
            already be selected when the prompt is opened. Note that only one of
            `default_indices` and `default_file` may be used. Defaults to [].
        default_file (Union[Path, str], optional): The name of a file to use for
            storing/loading what values should be chosen by default, or `None`
            if no such features is desired. Note that only one of`default_file`
            and `default_indices` may be used. Defaults to None.
        return_index (bool, optional): If `True`, the indices of chose items are
            returned instead of the items themselves. Defaults to False.
        allow_empty (bool, optional): If `True`, an empty selection is allowed.
            Defaults to False.

    Raises:
        ValueError: If both a `default_file` and `default_indices` were supplied.

    Returns:
        list[Any] | list[int]: The selected items, or their indices if `return_index=True`.
    """
    push_indent(1)

    if len(default_indices) > 0 and default_file is not None:
        raise ValueError(
            "It is not possible to use both default indices and default file."
        )

    # Get default selection
    if default_file is not None:
        strings = _load_strings(default_file)
        if strings is not None:
            default_indices = [list(options).index(s) for s in strings if s in options]

    indices = list(default_indices)

    while True:

        # Derive non-selected options
        remaining_indices = list(
            filter(lambda j: j not in indices, range(len(options)))
        )

        # Prompt directive
        print_with_border(directive, "*")

        # Print available items
        if len(remaining_indices) > 0:
            _print("Items to select from:")
            _print_opt_list([options[i] for i in remaining_indices], remaining_indices)
        else:
            _print("There are no more items to pick.")

        # Print current selection
        if len(indices) > 0:
            _print("Selected items:")
            _print_opt_list([options[i] for i in indices], indices)

        # Help prompt
        _print("(type 'h' for help)")

        # Receive input
        answer = _input("Pick / drop items")

        # Check if indices should be added or dropped
        drop = False
        if re.match(r"^d(rop)? .+$", answer):
            drop = True
            answer = answer[answer.find(" ") + 1 :]

        # Derive indices from input string
        selected_indices = _unravel_indices(answer)

        # Add / remove requested indices
        for selected_index in selected_indices:
            if drop:
                if selected_index in indices:
                    indices.remove(selected_index)
            else:
                if selected_index in remaining_indices:
                    indices.append(selected_index)
        indices.sort()

        # Check if done
        if _opt_match(answer, ["done"]):
            if len(indices) > 0 or allow_empty:
                break
            else:
                print_with_border("Must pick at least one option", "!")
                _input()
        elif _opt_match(answer, ("h", "help")):
            push_indent(2)

            print_with_border("Selection Help", "*")

            _print("* Enter numbers associated with the items you would like to pick.")
            _print("* Numbers may be specified on singular form, or as sequences.")
            _print(
                "* Multiple numbers or sequences may be specified at once if separate by a comma."
            )
            _print("* Examples: '5', '42-69', '13:17', '2-4,7:9,12'")
            _print(
                "* To unpick items, prefix the command with 'drop' (or 'd'), e.g., 'drop 5-6'."
            )
            _print("* Type 'done' to complete selection.")
            _input()

            pop_indent()

    pop_indent()

    # Save default indices if applicable
    if default_file is not None:
        _save_strings(default_file, [options[i] for i in indices])

    return indices if return_index else [options[i] for i in indices]
