import abc

from typing import Any, Callable, Union

from typing_extensions import Protocol


class SetupFunction(Protocol):
    def __call__(self, improve: bool, **setup_parameters: Any) -> None:
        ...


class SetupMode:
    """
    A wrapper for all functions associated with a setup mode.
    """

    def __init__(
        self,
        setup_func: SetupFunction,
        ready_func: Callable[[], bool],
        info_func: Callable[[], str] = lambda: "No information available.",
        required_modes: list[str] = [],
        **parameters: Any,
    ):
        """
        Constructs a new setup mode wrapper.

        Args:
            setup_func (SetupFunction): A function for performing the actual setup.
                When this function is called from `Setupable.setup()`, parameters will be forwarded to it.
                All valid parameters are expected to be specified by the `**parameters` argument. The
                `improve` flag indicates if the mode should improve its underlying models, or restart.
            ready_func (Callable[[], bool]): A function for checking whether a setup has been completed.
            info_func (Callable[[], Union[str, None]], optional): A function for fetching info about a completed setup.
            required_modes (list[str], optional): List of setup modes that must be completed before this mode.
            **parameters (Any): Setup parameters to pass to `setup_func`, and their default values.
        """
        self._setup_func = setup_func
        self._ready_func = ready_func
        self._info_func = info_func
        self._required_modes = required_modes
        self._parameters = parameters

    @property
    def setup(self) -> SetupFunction:
        """
        Function for performing the actual setup.
        """
        return self._setup_func

    @property
    def ready(self) -> Callable[[], bool]:
        """
        Function for checking whether the setup has been completed.
        """
        return self._ready_func

    @property
    def info(self) -> Callable[[], Union[str, None]]:
        """
        Function for fetching info about a completed setup.
        """
        return self._info_func

    @property
    def required_modes(self) -> list[str]:
        """
        List of setup modes that must be completed before this mode.
        """
        return self._required_modes

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Available setup parameters and their default values.
        """
        return self._parameters


class Setupable(metaclass=abc.ABCMeta):
    """
    Interface for classes that are supposed to be setupable.
    """

    SETUP_MODE_ALL = "all"
    SETUP_MODE_IMPROVE = "improve"
    SETUP_MODE_CONTINUE = "continue"

    def _check_valid_mode(self, mode: str, check_prerequisites: bool = False) -> None:

        # Check whether the mode exists
        if not mode in self.get_setup_modes():
            raise ValueError(
                f"Unrecognized setup mode: '{mode}'. Available modes are: {self.get_setup_modes()}"
            )

        # Check whether prerequisite modes have been completed (if applicable)
        if check_prerequisites:
            for m in self.reg_setup_modes()[mode].required_modes:
                if not self.is_ready(m):
                    raise ValueError(
                        f"Setup mode '{m}' must be ready before '{mode}'. Make sure that "
                        + f"all prerequisites of '{mode}' are completed before performing its setup."
                    )

    def _compile_ordered_setup_mode_list(
        self, mode: str = None, li: list[str] = None
    ) -> list[str]:
        if mode is None or li is None:
            li = []

            # Add all modes
            for m in self.get_setup_modes():
                self._compile_ordered_setup_mode_list(m, li)

            return li
        else:
            # Add prerequisites first
            for r in self.get_required_modes(mode):
                self._compile_ordered_setup_mode_list(r, li)

            # Add mode when prerequisites are satisfied
            if mode not in li:
                li.append(mode)

    def setup(
        self,
        mode: str,
        parameters: dict[str, Union[dict[str, Any], Any]] = {},
        skip_if_completed: bool = False,
        improve_if_completed: bool = False,
    ) -> None:
        """
        Runs the specified setup mode. Must be one of the specific modes
        registered by the setupable (see `get_setup_modes()`), or one of the
        "generic modes", i.e., `SETUP_MODE_ALL`, `SETUP_MODE_IMPROVE`, or
        `SETUP_MODE_CONTINUE`. Generic modes do not have their own
        implementations - instead, they execute multiple modes according to
        different rules. See the `mode` parameter for details.

        Note that if a specific mode is issued, all of its prerequisite setup modes must have
        been performed prior or else an exception will be raised. To see the prerequisites of
        a specific mode, use `get_required_modes(mode)`.

        Args:
            mode (str): The mode to perform a setup for. Either a specific mode, or one of
                the generic ones. If the mode is set to`SETUP_MODE_ALL`, all specific modes
                will be executed in declared order, with `improve=False`, i.e., indicating that
                underlying models should be retrained from scratch. The setup mode
                `SETUP_MODE_IMPROVE` is the same, except that `improve=True`, implying that that
                models should be improved upon by further training. If the mode is set to
                `SETUP_MODE_CONTINUE`, only the specific modes that are not yet ready will be executed.
            parameters (dict[str, Union[dict[str, Any], Any]], optional): Parameters to use
                for the specified setup mode. For a specific mode, the dictionary should
                contain parameter names mapped to desired values - omitted parameters will
                be replaced by their default values. If generic modes are issued, the
                dictionary should instead contain multiple dictionaries of the aforementioned
                format, but nested under specific mode names, i.e., there will be a dictionary
                of specific mode parameters for each mode. If default values should be
                used for all parameters of a specific mode (for the generic case), the mode
                may be omitted from the dictionary. To list available parameters and their
                default values for specific modes, use `get_setup_parameters(mode)`.
            skip_if_completed (bool, optional): If `True`, the specified `mode`
                will not be setup if it is already ready. This parameter has no
                effect for generic modes, e.g., `SETUP_MODE_ALL` or `SETUP_MODE_CONTINUE`.
            improve_if_completed (bool, optional): If `True`, the `improve` flag will be
                forwarded to the setup function if its mode is already ready. This parameter
                has no effect for generic modes, or if `skip_if_completed=True`.

        Raises:
            ValueError: If the specified setup mode is not recognized.
            ValueError: If parameters are specified for an unrecognized setup mode.
            ValueError: If prerequisite setup modes have not been completed.
            ValueError: If an unrecognized parameter was given.
            ValueError: If parameters are not specified per-mode when multiple setups
                are to be performed, i.e., for generic setup modes.
        """
        if mode in (
            self.SETUP_MODE_ALL,
            self.SETUP_MODE_IMPROVE,
            self.SETUP_MODE_CONTINUE,
        ):
            modes_to_setup = self._compile_ordered_setup_mode_list()

            # Parameter check
            if any(not isinstance(p, dict) for p in parameters.values()):
                raise ValueError(
                    "Invalid parameter format! Make sure that all modes have their own parameter setups."
                )

            # Mode check
            for m in parameters.keys():
                self._check_valid_mode(m)

            # Setup all modes
            for m in modes_to_setup:
                self.setup(
                    m,
                    parameters=parameters[m] if m in parameters else {},
                    skip_if_completed=mode == self.SETUP_MODE_CONTINUE,
                    improve_if_completed=mode == self.SETUP_MODE_IMPROVE,
                )
        else:
            # Check if mode exists
            self._check_valid_mode(mode, check_prerequisites=True)

            # Fetch mode
            sm = self.reg_setup_modes()[mode]

            # Skip if mode is completed (if applicable)
            if skip_if_completed and sm.ready():
                return

            # Check if all specified parameters exist
            for p in parameters.keys():
                if p not in sm.parameters.keys():
                    raise ValueError(
                        f"Unrecognized parameter '{p}' for mode '{mode}'. "
                        + f"Available parameters are: {list(self.get_setup_parameters(mode).keys())}"
                    )

            # Extract specified parameters and default values
            p = {
                pname: (
                    parameters[pname] if pname in parameters else sm.parameters[pname]
                )
                for pname in sm.parameters.keys()
            }

            # Print setup mode to CLI
            print("-" * (len(mode) + 24))
            print(f"| Running setup mode: {mode} |")
            print("-" * (len(mode) + 24))

            # Perform actual setup
            sm.setup(improve_if_completed and sm.ready(), **p)

    def get_setup_modes(self) -> list[str]:
        """
        Returns a list of all available setup modes. Note that the generic
        modes `SETUP_MODE_ALL` and `SETUP_MODE_CONTINUE` are excluded.

        Returns:
            list[str]: A list of all implementation-specific
                setup modes, analogous to the keys of `reg_setup_modes()`.
        """
        return list(self.reg_setup_modes().keys())

    def get_setup_parameters(self, mode: str) -> dict[str, Any]:
        """
        Returns all available parameters of the specified setup mode.

        Args:
            mode (str): The mode to fetch the parameters for.

        Returns:
            dict[str, Any]: Parameter names mapped to their default values.

        Raises:
            ValueError: If the specified setup mode is not recognized.
        """

        # Check if mode exists
        self._check_valid_mode(mode)

        # Fetch available parameters
        return self.reg_setup_modes()[mode].parameters

    def is_ready(self, mode: str = None) -> bool:
        """
        Checks whether the setup of the specified mode has been completed.

        Args:
            mode (str, optional): The mode to check for. If None, it will check
                if all modes have been completed. Defaults to None.

        Returns:
            bool: `True` if the setup of the specified mode has been completed.
                Note that if `mode=None`, it will instead return `True` if and
                only if all setups have been completed.

        Raises:
            ValueError: If the specified setup mode is not recognized.
        """

        # Check if all modes are ready
        if mode is None:
            return not any(not s.ready() for _, s in self.reg_setup_modes().items())

        # Check if mode exists
        self._check_valid_mode(mode)

        # Check if single mode is ready
        return self.reg_setup_modes()[mode].ready()

    def get_setup_info(self, mode: str) -> Union[str, None]:
        """
        Returns info about the specified setup.

        Args:
            mode (str): The mode to fetch info from.

        Returns:
            str: Information about a performed setup mode.
                If the setup has not been performed, `None` is returned instead.

        Raises:
            ValueError: If the specified setup mode is not recognized.
        """

        # Check if mode exists
        self._check_valid_mode(mode)

        # Fetch info
        s = self.reg_setup_modes()[mode]
        return s.info() if s.ready() else None

    def get_required_modes(self, mode: str) -> list[str]:
        """
        Returns a list of all prerequisites of the specified mode. All of these
        prerequisite modes must be completed before performing a setup of the
        specified mode.

        Args:
            mode (str): The mode whose prerequisites should be fetched.

        Returns:
            list[str]: A list of all prerequisites modes of the specified setup mode.
        """

        # Check if mode exists
        self._check_valid_mode(mode)

        # Fetch prerequisites
        return self.reg_setup_modes()[mode].required_modes

    @abc.abstractmethod
    def reg_setup_modes(self) -> dict[str, SetupMode]:
        """
        Should return a dictionary declaring all available setup modes.

        The dictionary should consist of setup mode names, mapped to `SetupMode`
        objects. The dictionary may be dynamically generate if desired.

        Returns:
            dict[str, SetupMode]: All setup modes, and their functionality.
        """
        pass
