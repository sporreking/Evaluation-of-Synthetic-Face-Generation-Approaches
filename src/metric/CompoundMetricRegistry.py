from src.metric.CompoundMetric import CompoundMetric
from src.core.Registry import Registry
from src.metric.FIDCompoundMetric import FIDCompoundMetric, FID_NAME
from src.metric.LSCompoundMetric import LSCompoundMetric, LS_NAME
from src.metric.PPLCompoundMetric import PPLCompoundMetric, PPL_NAME
from src.metric.RecallCompoundMetric import RecallCompoundMetric, RECALL_NAME
from src.metric.PrecisionCompoundMetric import PrecisionCompoundMetric, PRECISION_NAME


class CompoundMetricRegistry(Registry):
    """
    Static class implementing the abstract Registry class
    used for initialization and storing of all subclasses of the
    CompoundMetric class.

    * If more CompoundMetrics are implemented they must be manually added
    to the internal storage (`_COMPOUND_METRICS`) of this class.
    """

    _COMPOUND_METRICS = {
        FID_NAME: FIDCompoundMetric,
        LS_NAME: LSCompoundMetric,
        PPL_NAME: PPLCompoundMetric,
        PRECISION_NAME: PrecisionCompoundMetric,
        RECALL_NAME: RecallCompoundMetric,
    }

    @staticmethod
    def get_names() -> list[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(CompoundMetricRegistry._COMPOUND_METRICS.keys())

    @staticmethod
    def get_resource(name: str) -> type[CompoundMetric]:
        """
        Returns a compound metric with the given `name` from the registry.

        Args:
            name (str): Name of the CompoundMetric.

        Returns:
            CompoundMetric: CompoundMetric with the given `name`.
        """
        return CompoundMetricRegistry._COMPOUND_METRICS[name]

    @staticmethod
    def get_resources() -> list[type[CompoundMetric]]:
        return list(CompoundMetricRegistry._COMPOUND_METRICS.values())
