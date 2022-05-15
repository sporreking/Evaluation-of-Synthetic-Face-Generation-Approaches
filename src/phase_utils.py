from src.dataset.Dataset import Dataset
import src.util.PromptUtil as PU


def get_prompt_save_file_name(name: str) -> str:
    """
    Derives a file name to use for storing default prompt values.

    Args:
        name (str): The unique identifier of the file-name.

    Returns:
        str: The file name.
    """
    return f"{name}.sav"


def confirm_lists(
    message: str, *lists: tuple[str, list[str]], confirm_phrase: str = "Confirm?"
) -> bool:
    """
    Displays a number of lists, each with their own header, and asks the user
    whether they would like to confirm what has been displayed.

    Args:
        message (str): The main message to present, headering all of the lists.
        *lists (tuple[str, list[str]]): Lists to present, coupled with proceding headers.
        confirm_phrase (str, optional): Message for the yes/no prompt. Default is "Confirm?".
    Returns:
        bool: `True` if the user confirms the lists, otherwise `False`.
    """
    PU.push_indent(1)
    PU.print_with_border(message, "=", "||")

    for header, li in lists:
        PU.print_list(
            header,
            li,
            header_border_symbol="-",
            header_border_side_symbol="|",
        )

    accept_selection = PU.prompt_yes_no("Confirm?")

    PU.pop_indent()

    return accept_selection
