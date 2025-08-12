from src.plot import plot_model
from src.model import MultiTargetMultiEquation_HSModel
from src.model import MultiTargetMultiEquation_Normal

plot_model(MultiTargetMultiEquation_Normal,"Normal")
plot_model(MultiTargetMultiEquation_HSModel,"BH")