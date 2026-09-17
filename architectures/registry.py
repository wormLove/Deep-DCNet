from architectures.single_layer import BiologicalClassifier
from architectures.stacked import StackedBiologicalClassifier

# name -> architecture class. An experiment/training script can pick one by
# string instead of importing a specific class directly, matching the same
# registry pattern used in DC_Net_Hebbian_Model's architectures/registry.py.
ARCHITECTURES = {
    "single_layer": BiologicalClassifier,
    "stacked": StackedBiologicalClassifier,
}
