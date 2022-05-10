from src.metric.SampleMetric import SampleMetric
from src.core.Registry import Registry
from src.metric.DatasetSimilaritySampleMetric import (
    DatasetSimilaritySampleMetric,
    DATASET_SIMILARITY_NAME,
)
from src.metric.PopulationSimilaritySampleMetric import (
    PopulationSimilaritySampleMetric,
    POPULATION_SIMILARITY_NAME,
)


class SampleMetricRegistry(Registry):
    """
    Registry of available sample metrics.

    * If more SampleMetrics are implemented they must be manually added
    to the internal storage (`_SAMPLE_METRICS`) of this class.
    """

    _SAMPLE_METRICS = {
        DATASET_SIMILARITY_NAME: DatasetSimilaritySampleMetric,
        POPULATION_SIMILARITY_NAME: PopulationSimilaritySampleMetric,
    }

    @staticmethod
    def get_names() -> list[str]:
        """
        Returns all the names (keys) in the registry.

        Returns:
            list[str]: List of all names (keys) in the registry.
        """
        return list(SampleMetricRegistry._SAMPLE_METRICS.keys())

    @staticmethod
    def get_resource(name: str) -> type[SampleMetric]:
        """
        Returns a sample metric with the given `name` from the registry.

        Args:
            name (str): Name of the SampleMetric.

        Returns:
            SampleMetric: SampleMetric with the given `name`.
        """
        return SampleMetricRegistry._SAMPLE_METRICS[name]

    @staticmethod
    def get_resources() -> list[type[SampleMetric]]:
        return list(SampleMetricRegistry._SAMPLE_METRICS.values())
