import os
import shutil
import pickle
from manifold_mor.chart.auto_encoder import AutoEncoderChart
from manifold_mor.context import ManifoldMorContext
from manifold_mor.experiments.caching import PATH_CACHED_RESULTS
from manifold_mor_demos.hamiltonian_wave.manifold_experiments import (
    RED_DIMS, BumpA0AutoEncoderManifoldExperiment, BumpA1AutoEncoderManifoldExperiment, BumpAutoEncoderBasedManifoldExperiment, BumpLinearSubspaceExperiment)

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if "demos.mor" in module:
            renamed_module = module.replace("demos.mor", "manifold_mor_demos")

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def resave_with_renamed_modules(exp: BumpAutoEncoderBasedManifoldExperiment):
    path= os.path.join(
        exp.get_experiment_folder(),
        PATH_CACHED_RESULTS,
        '_compute_manifold.pkl',
    )
    path_copy = os.path.join(
        exp.get_experiment_folder(),
        PATH_CACHED_RESULTS,
        '_compute_manifold_old.pkl',
    )
    shutil.copy(
        path,
        path_copy
    )
    # load object with modified paths
    with open(path_copy, 'rb') as f:
        obj = RenameUnpickler(f).load()
    
    # save modified object
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    
    # test loading
    with open(path, 'rb') as f:
        obj = pickle.load(f)


if __name__ == '__main__':
    context = ManifoldMorContext()
    for red_dim in RED_DIMS:
        exp_0_s = BumpA0AutoEncoderManifoldExperiment(
            0.9,
            red_dim,
            shift=AutoEncoderChart.SHIFT_INITIAL_VALUE,
        )
        resave_with_renamed_modules(exp_0_s)
        exp_0 = BumpA0AutoEncoderManifoldExperiment(1., red_dim)
        resave_with_renamed_modules(exp_0)
        exp_1 = BumpA1AutoEncoderManifoldExperiment(red_dim)
        resave_with_renamed_modules(exp_1)
        exp_pca = BumpLinearSubspaceExperiment(red_dim, 'pca')
        resave_with_renamed_modules(exp_pca)
        exp_cotan = BumpLinearSubspaceExperiment(red_dim, 'psd_cotan')
        resave_with_renamed_modules(exp_cotan)