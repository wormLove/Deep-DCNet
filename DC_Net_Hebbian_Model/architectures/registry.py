from architectures.single_layer import BiologicalClassifier
from architectures.stacked import StackedBiologicalClassifier

# name -> architecture class. An experiment file picks one by string instead
# of importing a specific class directly, so adding a new architecture never
# means touching the training loop.
ARCHITECTURES = {
    "single_layer": BiologicalClassifier,
    "stacked": StackedBiologicalClassifier,
}
