'''
Wave model discussed in Buchfink, Glas, Haasdonk 2021
'''
from typing import List

import numpy as np
from manifold_mor.experiments.model import ModelExperiment
from manifold_mor.models.hamiltonian import TravellingBumpModel
from manifold_mor.time_steppers.hamiltonian import HamiltonianModelTimeStepper

class BumpModelExperiment(ModelExperiment):
    def __init__(
        self,
        mu_scenario: str,
        shifted: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(
            mu_scenario,
            n_x=4*512+2,
            shifted=shifted,
            num_sampling=8,
            t_0=0, dt=1/4000, t_end=1.,
            max_iter_newton = 15,
            debug=debug,
        )
        self.modified_parameter_keys.update(['mu_scenario', 'shifted'])
        if debug:
            self.modified_parameter_keys.add('debug')
            # fewer time steps, parameters and lower spatial resolution
            self._parameters['t_end'] = 50*self['dt']
            self._parameters['num_sampling'] = 2

    def get_experiment_name(self) -> str:
        return 'bump'

    def get_model(self) -> TravellingBumpModel:
        model = TravellingBumpModel(HamiltonianModelTimeStepper.INTEGRATOR_IMPL_MIDPOINT, self['dt'], n_x=self['n_x'])
        if self['shifted']:
            model = model.to_shifted()
        model.time_stepper.non_linear_solver.max_iter = self['max_iter_newton']
        return model

    def get_mus(self) -> List[dict]:
        mu_scenario = self['mu_scenario']
        num_waves = 4
        q0_supp = 1 / (num_waves+2)
        max_speed = (1 - q0_supp) / (self['t_end'] - self['t_0'])
        min_speed = max_speed / 2
        if mu_scenario in [ModelExperiment.MU_TRAINING, ModelExperiment.MU_TEST_REPRODUCTION]:
            num_sampling = self['num_sampling']
            wave_speed = np.linspace(min_speed, max_speed, num=num_sampling)
            assert num_sampling % 2 == 0, 'num_sampling has to be even, otherwise generalization case is reproduction'
        elif mu_scenario == ModelExperiment.MU_TEST_GENERALIZATION:
            wave_speed = [0.51, 0.625, 0.74] # rounded interval centers
            if self['debug']:
                # only one mus
                wave_speed = wave_speed[:1]
        else:
            raise NotImplementedError('Unknown mu_scenario "{}".'.format(mu_scenario))
        return [{'mu_scenario': mu_scenario, 'mu_idx': idx, 'c': c, 'q0_supp': q0_supp * c / max_speed} for idx, c in enumerate(wave_speed)]
