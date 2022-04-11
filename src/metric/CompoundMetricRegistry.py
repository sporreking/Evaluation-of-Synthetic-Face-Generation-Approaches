from src.metric.CompoundMetric import CompoundMetric
from src.core.Registry import Registry


class CompoundMetricRegistry(Registry):
    """
    Static class implementing the abstract Registry class
    used for initialization and storing of all subclasses of the
    CompoundMetric class.

    * If more CompoundMetrics are implemented they must be manually added
    to the internal storage (`_COMPOUND_METRICS`) of this class.
    """

    _COMPOUND_METRICS = {}

    @staticmethod
    def get_names() -> list[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(CompoundMetricRegistry._COMPOUND_METRICS.keys())

    @staticmethod
    def get_resource(name: str) -> CompoundMetric:
        """
        Returns a compound metric with the given `name` from the registry.

        Args:
            name (str): Name of the CompoundMetric.

        Returns:
            CompoundMetric: CompoundMetric with the given `name`.
        """
        return CompoundMetricRegistry._COMPOUND_METRICS[name]

    @staticmethod
    def get_resources() -> list[CompoundMetric]:
        return list(CompoundMetricRegistry._COMPOUND_METRICS.values())
