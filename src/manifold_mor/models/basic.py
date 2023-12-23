'''General definition of a model.'''
from abc import ABC, abstractmethod
from manifold_mor.fields.vector import VectorField

from manifold_mor.time_steppers.basic import TimeStepper


class Model(ABC):
    def __init__(self, name: str, time_stepper: TimeStepper):
        assert isinstance(time_stepper, TimeStepper)
        assert isinstance(name, str)
        self.time_stepper = time_stepper
        self.name = name

    def solve(self, t_0, t_end, dt, mu, logger=None, hook_fcns=None):
        assert self.check_mu(mu)
        if logger:
            # start and end run if no active run is registered
            if not logger.has_active_run():
                logger.start_run()
                end_run = True
            else:
                end_run = False
            logger.log_params(mu)
            logger.log_params({'t_0': t_0, 't_end': t_end, 'dt': dt})
            callbacks = (logger.report_time_stepper_callback,)
        else:
            callbacks = None
        time_stepper_result = self.time_stepper.solve(t_0, t_end, dt, mu, hook_fcns=hook_fcns, callbacks=callbacks)
        if logger and end_run:
            logger.end_run()
        return time_stepper_result

    def to_shifted(self):
        from manifold_mor.models.shifted import ShiftedModel
        return ShiftedModel(self)

    @abstractmethod
    def initial_value(self, mu):
        ...

    @abstractmethod
    def check_mu(self, mu):
        ...

    @abstractmethod
    def get_dim(self):
        ...

    def set_mu(self, mu):
        if self.time_stepper.is_for_vector_field():
            self.vector_field.set_mu(mu)
        elif self.time_stepper.is_for_covector_field():
            self.metric.set_mu(mu)
            self.covector_field.set_mu(mu)
        else:
            raise NotImplementedError()

    def compute_rhs(self, x):
        if hasattr(self, 'vector_field') and isinstance(self.vector_field, VectorField):
            return self.vector_field.eval(x)
        else:
            raise NotImplementedError()
