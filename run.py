from src.dataset.DatasetRegistry import DatasetRegistry
from src.population.Population import Population
from src.dataset.Dataset import Dataset

# Load dataset
DS: Dataset = DatasetRegistry.get_resources()[0]()
print(f"Using dataset: {DS.get_name(DS.get_resolution())}")

# Load population
POP = Population("test")
print(f"Using population: {POP.get_name()}")

import src.metric.EvaluationEmbedding as EE
from src.util.CudaUtil import *
import torch
from src.metric.AlphaPrecisionSampleMetricDescription import (
    AlphaPrecisionSampleMetricDescription as AlphaPrecisionSMD,
)

# Inception model
# device = get_default_device()

# Model
# model = to_device(EE.DeepSVDDNet(), device)

# Sample metric manager
from src.metric.SampleMetricManager import SampleMetricManager

smm = SampleMetricManager([AlphaPrecisionSMD], POP, DS)

# Torch dataset
# TDS = DS.to_torch_dataset(EE.get_inception_image_transform(), use_labels=False)
# loader = torch.utils.data.DataLoader(TDS, batch_size=2, shuffle=False, num_workers=2)

c = input("Run inception model, setup, project, or calc?(i,s,p,c)")
if c == "i":
    """b = to_device(next(iter(loader)), device)
    print(b.shape)
    r = model.inception(b)
    print(r)
    print(r.shape)"""
elif c == "s":
    AlphaPrecisionSMD.setup(DS, "continue")
elif c == "p":
    AlphaPrecisionSMD.setup(DS, "project")
elif c == "poptest2":
    import numpy as np
    from src.util.ZeroPaddedIterator import zero_padded_iterator

    # Create population
    p = Population("test2")
    N = 1513

    p.add_all(
        np.round(np.random.rand(N, 4), 6),  # Latent codes
        np.round(np.random.rand(N, 4), 6),  # Seed codes
        [f"population/test2/{i}.png" for i in zero_padded_iterator(0, N, 6)],  # URIs
        [np.random.randint(128) for _ in range(N)],  # Filters
        append=False,
        save_to_disk=True,  # Save to disk
        age=[np.random.randint(100) for _ in range(N)],  # Attribute 1
        smile=[np.random.randint(2) == 0 for _ in range(N)],  # Attribute 2
    )
elif c == "c":
    smm.calc(AlphaPrecisionSMD.get_name(), alpha=float(input("Alpha? ")))
    res = smm.get(AlphaPrecisionSMD.get_name())

    print(res)
    print(sum(list(res["AlphaPrecision"])))
else:
    exit()
